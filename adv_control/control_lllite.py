# adapted from https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI
# basically, all the LLLite core code is from there, which I then combined with
# Advanced-ControlNet features and QoL
import math
from typing import Union
from torch import Tensor
import torch
import os

import comfy.utils
import comfy.ops
import comfy.model_management
from comfy.model_patcher import ModelPatcher
from comfy.controlnet import ControlBase

from .logger import logger
from .utils import (AdvancedControlBase, TimestepKeyframeGroup, ControlWeights, broadcast_image_to_extend, extend_to_batch_size,
                    prepare_mask_batch)


# based on set_model_patch code in comfy/model_patcher.py
def set_model_patch(transformer_options, patch, name):
    to = transformer_options
    # check if patch was already added
    if "patches" in to:
        current_patches = to["patches"].get(name, [])
        if patch in current_patches:
            return
    if "patches" not in to:
        to["patches"] = {}
    to["patches"][name] = to["patches"].get(name, []) + [patch]

def set_model_attn1_patch(transformer_options, patch):
    set_model_patch(transformer_options, patch, "attn1_patch")

def set_model_attn2_patch(transformer_options, patch):
    set_model_patch(transformer_options, patch, "attn2_patch")


def extra_options_to_module_prefix(extra_options):
    # extra_options = {'transformer_index': 2, 'block_index': 8, 'original_shape': [2, 4, 128, 128], 'block': ('input', 7), 'n_heads': 20, 'dim_head': 64}

    # block is: [('input', 4), ('input', 5), ('input', 7), ('input', 8), ('middle', 0),
    #   ('output', 0), ('output', 1), ('output', 2), ('output', 3), ('output', 4), ('output', 5)]
    # transformer_index is: [0, 1, 2, 3, 4, 5, 6, 7, 8], for each block
    # block_index is: 0-1 or 0-9, depends on the block
    # input 7 and 8, middle has 10 blocks

    # make module name from extra_options
    block = extra_options["block"]
    block_index = extra_options["block_index"]
    if block[0] == "input":
        module_pfx = f"lllite_unet_input_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    elif block[0] == "middle":
        module_pfx = f"lllite_unet_middle_block_1_transformer_blocks_{block_index}"
    elif block[0] == "output":
        module_pfx = f"lllite_unet_output_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    else:
        raise Exception(f"ControlLLLite: invalid block name '{block[0]}'. Expected 'input', 'middle', or 'output'.")
    return module_pfx


class LLLitePatch:
    ATTN1 = "attn1"
    ATTN2 = "attn2"
    def __init__(self, modules: dict[str, 'LLLiteModule'], patch_type: str, control: Union[AdvancedControlBase, ControlBase]=None):
        self.modules = modules
        self.control = control
        self.patch_type = patch_type
        #logger.error(f"create LLLitePatch: {id(self)},{control}")
    
    def __call__(self, q, k, v, extra_options):
        #logger.error(f"in __call__: {id(self)}")
        # determine if have anything to run
        if self.control.timestep_range is not None:
            # it turns out comparing single-value tensors to floats is extremely slow
            # a: Tensor = extra_options["sigmas"][0]
            if self.control.t > self.control.timestep_range[0] or self.control.t < self.control.timestep_range[1]:
                return q, k, v

        module_pfx = extra_options_to_module_prefix(extra_options)

        is_attn1 = q.shape[-1] == k.shape[-1]  # self attention
        if is_attn1:
            module_pfx = module_pfx + "_attn1"
        else:
            module_pfx = module_pfx + "_attn2"

        module_pfx_to_q = module_pfx + "_to_q"
        module_pfx_to_k = module_pfx + "_to_k"
        module_pfx_to_v = module_pfx + "_to_v"

        if module_pfx_to_q in self.modules:
            q = q + self.modules[module_pfx_to_q](q, self.control)
        if module_pfx_to_k in self.modules:
            k = k + self.modules[module_pfx_to_k](k, self.control)
        if module_pfx_to_v in self.modules:
            v = v + self.modules[module_pfx_to_v](v, self.control)

        return q, k, v

    def to(self, device):
        #logger.info(f"to... has control? {self.control}")
        for d in self.modules.keys():
            self.modules[d] = self.modules[d].to(device)
        return self
    
    def set_control(self, control: Union[AdvancedControlBase, ControlBase]) -> 'LLLitePatch':
        self.control = control
        return self
        #logger.error(f"set control for LLLitePatch: {id(self)}, cn: {id(control)}")

    def clone_with_control(self, control: AdvancedControlBase):
        #logger.error(f"clone-set control for LLLitePatch: {id(self)},{id(control)}")
        return LLLitePatch(self.modules, self.patch_type, control)

    def cleanup(self):
        for module in self.modules.values():
            module.cleanup()


# TODO: use comfy.ops to support fp8 properly
class LLLiteModule(torch.nn.Module):
    def __init__(
        self,
        name: str,
        is_conv2d: bool,
        in_dim: int,
        depth: int,
        cond_emb_dim: int,
        mlp_dim: int,
    ):
        super().__init__()
        self.name = name
        self.is_conv2d = is_conv2d
        self.is_first = False

        modules = []
        modules.append(torch.nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))  # to latent (from VAE) size*2
        if depth == 1:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))
        elif depth == 2:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0))
        elif depth == 3:
            # kernel size 8 is too large, so set it to 4
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))

        self.conditioning1 = torch.nn.Sequential(*modules)

        if self.is_conv2d:
            self.down = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim + cond_emb_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim, in_dim, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.down = torch.nn.Sequential(
                torch.nn.Linear(in_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim + cond_emb_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim, in_dim),
            )

        self.depth = depth
        self.cond_emb = None
        self.cx_shape = None
        self.prev_batch = 0
        self.prev_sub_idxs = None

    def cleanup(self):
        del self.cond_emb
        self.cond_emb = None
        self.cx_shape = None
        self.prev_batch = 0
        self.prev_sub_idxs = None

    def forward(self, x: Tensor, control: Union[AdvancedControlBase, ControlBase]):
        mask = None
        mask_tk = None
        #logger.info(x.shape)
        if self.cond_emb is None or control.sub_idxs != self.prev_sub_idxs or x.shape[0] != self.prev_batch:
            # print(f"cond_emb is None, {self.name}")
            cond_hint = control.cond_hint.to(x.device, dtype=x.dtype)
            if control.latent_dims_div2 is not None and x.shape[-1] != 1280:
                cond_hint = comfy.utils.common_upscale(cond_hint, control.latent_dims_div2[0] * 8, control.latent_dims_div2[1] * 8, 'nearest-exact', "center").to(x.device, dtype=x.dtype)
            elif control.latent_dims_div4 is not None and x.shape[-1] == 1280:
                cond_hint = comfy.utils.common_upscale(cond_hint, control.latent_dims_div4[0] * 8, control.latent_dims_div4[1] * 8, 'nearest-exact', "center").to(x.device, dtype=x.dtype)
            cx = self.conditioning1(cond_hint)
            self.cx_shape = cx.shape
            if not self.is_conv2d:
                # reshape / b,c,h,w -> b,h*w,c
                n, c, h, w = cx.shape
                cx = cx.view(n, c, h * w).permute(0, 2, 1)
            self.cond_emb = cx
        # save prev values
        self.prev_batch = x.shape[0]
        self.prev_sub_idxs = control.sub_idxs

        cx: torch.Tensor = self.cond_emb
        # print(f"forward {self.name}, {cx.shape}, {x.shape}")

        # TODO: make masks work for conv2d (could not find any ControlLLLites at this time that use them)
        # create masks
        if not self.is_conv2d:
            n, c, h, w = self.cx_shape
            if control.mask_cond_hint is not None:
                mask = prepare_mask_batch(control.mask_cond_hint, (1, 1, h, w)).to(cx.dtype)
                mask = mask.view(mask.shape[0], 1, h * w).permute(0, 2, 1)
            if control.tk_mask_cond_hint is not None:
                mask_tk = prepare_mask_batch(control.mask_cond_hint, (1, 1, h, w)).to(cx.dtype)
                mask_tk = mask_tk.view(mask_tk.shape[0], 1, h * w).permute(0, 2, 1)

        # x in uncond/cond doubles batch size
        if x.shape[0] != cx.shape[0]:
            if self.is_conv2d:
                cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1, 1)
            else:
                # print("x.shape[0] != cx.shape[0]", x.shape[0], cx.shape[0])
                cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)
                if mask is not None:
                    mask = mask.repeat(x.shape[0] // mask.shape[0], 1, 1)
                if mask_tk is not None:
                    mask_tk = mask_tk.repeat(x.shape[0] // mask_tk.shape[0], 1, 1)

        if mask is None:
            mask = 1.0
        elif mask_tk is not None:
            mask = mask * mask_tk

        #logger.info(f"cs: {cx.shape}, x: {x.shape}, is_conv2d: {self.is_conv2d}")
        cx = torch.cat([cx, self.down(x)], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)
        cx = self.up(cx)
        if control.latent_keyframes is not None:
            cx = cx * control.calc_latent_keyframe_mults(x=cx, batched_number=control.batched_number)
        if control.weights is not None and control.weights.has_uncond_multiplier:
            cond_or_uncond = control.batched_number.cond_or_uncond
            actual_length = cx.size(0) // control.batched_number
            for idx, cond_type in enumerate(cond_or_uncond):
                # if uncond, set to weight's uncond_multiplier
                if cond_type == 1:
                    cx[actual_length*idx:actual_length*(idx+1)] *= control.weights.uncond_multiplier
        return cx * mask * control.strength * control._current_timestep_keyframe.strength


class ControlLLLiteModules(torch.nn.Module):
    def __init__(self, patch_attn1: LLLitePatch, patch_attn2: LLLitePatch):
        super().__init__()
        self.patch_attn1_modules = torch.nn.Sequential(*list(patch_attn1.modules.values()))
        self.patch_attn2_modules = torch.nn.Sequential(*list(patch_attn2.modules.values()))


class ControlLLLiteAdvanced(ControlBase, AdvancedControlBase):
    # This ControlNet is more of an attention patch than a traditional controlnet
    def __init__(self, patch_attn1: LLLitePatch, patch_attn2: LLLitePatch, timestep_keyframes: TimestepKeyframeGroup, device, ops: comfy.ops.disable_weight_init):
        super().__init__()
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controllllite())
        self.device = device
        self.ops = ops
        self.patch_attn1 = patch_attn1.clone_with_control(self)
        self.patch_attn2 = patch_attn2.clone_with_control(self)
        self.control_model = ControlLLLiteModules(self.patch_attn1, self.patch_attn2)
        self.control_model_wrapped = ModelPatcher(self.control_model, load_device=device, offload_device=comfy.model_management.unet_offload_device())
        self.latent_dims_div2 = None
        self.latent_dims_div4 = None

    def set_cond_hint_inject(self, *args, **kwargs):
        to_return = super().set_cond_hint_inject(*args, **kwargs)
        # cond hint for LLLite needs to be scaled between (-1, 1) instead of (0, 1)
        self.cond_hint_original = self.cond_hint_original * 2.0 - 1.0
        return to_return

    def pre_run_advanced(self, *args, **kwargs):
        AdvancedControlBase.pre_run_advanced(self, *args, **kwargs)
        #logger.error(f"in cn: {id(self.patch_attn1)},{id(self.patch_attn2)}")
        self.patch_attn1.set_control(self)
        self.patch_attn2.set_control(self)
        #logger.warn(f"in pre_run_advanced: {id(self)}")
    
    def get_control_advanced(self, x_noisy: Tensor, t, cond, batched_number: int, transformer_options: dict):
        # normal ControlNet stuff
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return control_prev
        
        dtype = x_noisy.dtype
        # prepare cond_hint
        if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            # if self.cond_hint_original length greater or equal to real latent count, subdivide it before scaling
            if self.sub_idxs is not None:
                actual_cond_hint_orig = self.cond_hint_original
                if self.cond_hint_original.size(0) < self.full_latent_length:
                    actual_cond_hint_orig = extend_to_batch_size(tensor=actual_cond_hint_orig, batch_size=self.full_latent_length)
                self.cond_hint = comfy.utils.common_upscale(actual_cond_hint_orig[self.sub_idxs], x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype).to(x_noisy.device)
            else:
                self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype).to(x_noisy.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to_extend(self.cond_hint, x_noisy.shape[0], batched_number)
        # some special logic here compared to other controlnets:
        # * The cond_emb in attn patches will divide latent dims by 2 or 4, integer
        # * Due to this loss, the cond_emb will become smaller than x input if latent dims are not divisble by 2 or 4
        divisible_by_2_h = x_noisy.shape[2]%2==0
        divisible_by_2_w = x_noisy.shape[3]%2==0
        if not (divisible_by_2_h and divisible_by_2_w):
            #logger.warn(f"{x_noisy.shape} not divisible by 2!")
            new_h = (x_noisy.shape[2]//2)*2
            new_w = (x_noisy.shape[3]//2)*2
            if not divisible_by_2_h:
                new_h += 2
            if not divisible_by_2_w:
                new_w += 2
            self.latent_dims_div2 = (new_h, new_w)
        divisible_by_4_h = x_noisy.shape[2]%4==0
        divisible_by_4_w =  x_noisy.shape[3]%4==0
        if not (divisible_by_4_h and divisible_by_4_w):
            #logger.warn(f"{x_noisy.shape} not divisible by 4!")
            new_h = (x_noisy.shape[2]//4)*4
            new_w = (x_noisy.shape[3]//4)*4
            if not divisible_by_4_h:
                new_h += 4
            if not divisible_by_4_w:
                new_w += 4
            self.latent_dims_div4 = (new_h, new_w)
        # prepare mask
        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number)
        # done preparing; model patches will take care of everything now
        set_model_attn1_patch(transformer_options, self.patch_attn1.set_control(self))
        set_model_attn2_patch(transformer_options, self.patch_attn2.set_control(self))
        # return normal controlnet stuff
        return control_prev
    
    def get_models(self):
        to_return: list = super().get_models()
        to_return.append(self.control_model_wrapped)
        return to_return

    def cleanup_advanced(self):
        super().cleanup_advanced()
        self.patch_attn1.cleanup()
        self.patch_attn2.cleanup()
        self.latent_dims_div2 = None
        self.latent_dims_div4 = None
    
    def copy(self):
        c = ControlLLLiteAdvanced(self.patch_attn1, self.patch_attn2, self.timestep_keyframes, self.device, self.ops)
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c


def load_controllllite(ckpt_path: str, controlnet_data: dict[str, Tensor]=None, timestep_keyframe: TimestepKeyframeGroup=None):
    if controlnet_data is None:
        controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    # adapted from https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI
    # first, split weights for each module
    module_weights = {}
    for key, value in controlnet_data.items():
        fragments = key.split(".")
        module_name = fragments[0]
        weight_name = ".".join(fragments[1:])

        if module_name not in module_weights:
            module_weights[module_name] = {}
        module_weights[module_name][weight_name] = value

    unet_dtype = comfy.model_management.unet_dtype()
    load_device = comfy.model_management.get_torch_device()
    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
    ops = comfy.ops.disable_weight_init
    if manual_cast_dtype is not None:
        ops = comfy.ops.manual_cast

    # next, load each module
    modules = {}
    for module_name, weights in module_weights.items():
        # kohya planned to do something about how these should be chosen, so I'm not touching this
        # since I am not familiar with the logic for this
        if "conditioning1.4.weight" in weights:
            depth = 3
        elif weights["conditioning1.2.weight"].shape[-1] == 4:
            depth = 2
        else:
            depth = 1

        module = LLLiteModule(
            name=module_name,
            is_conv2d=weights["down.0.weight"].ndim == 4,
            in_dim=weights["down.0.weight"].shape[1],
            depth=depth,
            cond_emb_dim=weights["conditioning1.0.weight"].shape[0] * 2,
            mlp_dim=weights["down.0.weight"].shape[0],
        )
        # load weights into module
        module.load_state_dict(weights)
        modules[module_name] = module.to(dtype=unet_dtype)
        if len(modules) == 1:
            module.is_first = True

    #logger.info(f"loaded {ckpt_path} successfully, {len(modules)} modules")

    patch_attn1 = LLLitePatch(modules=modules, patch_type=LLLitePatch.ATTN1)
    patch_attn2 = LLLitePatch(modules=modules, patch_type=LLLitePatch.ATTN2)
    control = ControlLLLiteAdvanced(patch_attn1=patch_attn1, patch_attn2=patch_attn2, timestep_keyframes=timestep_keyframe, device=load_device, ops=ops)
    return control
