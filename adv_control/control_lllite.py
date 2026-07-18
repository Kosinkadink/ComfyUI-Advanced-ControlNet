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
import comfy.model_patcher
from comfy.model_patcher import ModelPatcher
from comfy.controlnet import ControlBase

try:
    import comfy.ldm.anima.lllite as comfy_anima_lllite
except ImportError:
    comfy_anima_lllite = None

from .logger import logger
from .utils import (AdvancedControlBase, TimestepKeyframeGroup, ControlWeights, broadcast_image_to_extend, extend_to_batch_size,
                    prepare_mask_batch)


class AnimaLLLiteConst:
    INPAINT_MASK = "anima_lllite_inpaint_mask"


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
                mask_tk = prepare_mask_batch(control.tk_mask_cond_hint, (1, 1, h, w)).to(cx.dtype)
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
        if mask_tk is not None:
            mask = mask * mask_tk

        #logger.info(f"cs: {cx.shape}, x: {x.shape}, is_conv2d: {self.is_conv2d}")
        cx = torch.cat([cx, self.down(x)], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)
        cx = self.up(cx)
        if control.latent_keyframes is not None:
            cx = cx * control.calc_latent_keyframe_mults(x=cx, batched_number=control.batched_number)
        if control.weights is not None and control.weights.has_uncond_multiplier:
            cond_or_uncond = control.cond_or_uncond
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


class AnimaLLLiteAdvancedPatch:
    def __init__(self, model_patch, control: 'AnimaLLLiteAdvanced'=None):
        self.model_patch = model_patch
        self.control = control

    def set_control(self, control: 'AnimaLLLiteAdvanced') -> 'AnimaLLLiteAdvancedPatch':
        self.control = control
        return self

    def clone_with_control(self, control: 'AnimaLLLiteAdvanced') -> 'AnimaLLLiteAdvancedPatch':
        return AnimaLLLiteAdvancedPatch(self.model_patch, control)

    def __call__(self, args):
        if not self.control.should_run():
            return args

        x = args["x"]
        if x.shape[2] != 1:
            raise ValueError(f"Anima LLLite only supports T=1, got T={x.shape[2]}")

        target_height = x.shape[-2] * 8
        target_width = x.shape[-1] * 8
        image = self.control.prepare_batched_tensor(self.control.cond_hint_original, x.shape[0])[:, :3]
        image = comfy.utils.common_upscale(image, target_width, target_height, "bicubic", crop="center").clamp(0.0, 1.0)
        image = image.to(device=x.device, dtype=x.dtype) * 2.0 - 1.0

        if self.model_patch.model.cond_in_channels == 4:
            mask = self.control.weights.extras.get(AnimaLLLiteConst.INPAINT_MASK)
            if mask is None:
                raise ValueError(
                    "Anima LLLite inpainting models require an inpaint mask. Connect a MASK to Anima LLLite Extras, "
                    "connect its cn_extras output to Default Weights, then connect CN_WEIGHTS to weights_override on Apply Advanced ControlNet."
                )
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.ndim != 4 or mask.shape[1] != 1:
                raise ValueError(f"Anima LLLite mask must have one channel, got shape {tuple(mask.shape)}")
            if image.shape[0] > 1:
                mask = self.control.prepare_batched_tensor(mask, image.shape[0], except_one=False)
            mask = comfy.utils.common_upscale(mask.float(), target_width, target_height, "nearest-exact", crop="center")
            if mask.shape[0] != image.shape[0]:
                if image.shape[0] % mask.shape[0] != 0:
                    raise ValueError(f"Anima LLLite mask batch {mask.shape[0]} cannot be broadcast to image batch {image.shape[0]}")
                mask = mask.repeat(image.shape[0] // mask.shape[0], 1, 1, 1)
            mask = (mask >= 0.5).to(device=x.device, dtype=x.dtype)
            if self.model_patch.model.inpaint_masked_input:
                image = image * (mask < 0.5).to(image.dtype)
            image = torch.cat((image, mask * 2.0 - 1.0), dim=1)

        cond_emb = self.model_patch.model.encode_conditioning(image)
        multiplier, weight_mask = self.control.prepare_multiplier(args["img"])
        args["transformer_options"]["model_patch_data"][self] = (cond_emb, multiplier, weight_mask)
        return args

    def to(self, device_or_dtype):
        return self

    def models(self):
        return [self.model_patch]


class AnimaLLLiteAdvancedAttentionPatch:
    def __init__(self, patch: AnimaLLLiteAdvancedPatch, targets):
        self.patch = patch
        self.targets = targets

    def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        patch_data = extra_options["model_patch_data"].get(self.patch)
        if patch_data is None:
            return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

        cond_emb, multiplier, weight_mask = patch_data
        block_index = extra_options["block_index"]
        strength = self.patch.control.get_block_strength(block_index, multiplier, weight_mask)
        values = {"q": q, "k": k, "v": v}
        for value_name, target in self.targets.items():
            values[value_name] = self.patch.model_patch.model.apply(values[value_name], cond_emb, block_index, target, strength)
        return {"q": values["q"], "k": values["k"], "v": values["v"], "pe": pe, "attn_mask": attn_mask}


class AnimaLLLiteAdvancedMLPPatch:
    def __init__(self, patch: AnimaLLLiteAdvancedPatch):
        self.patch = patch

    def __call__(self, args):
        patch_data = args["transformer_options"]["model_patch_data"].get(self.patch)
        if patch_data is None:
            return args

        cond_emb, multiplier, weight_mask = patch_data
        block_index = args["transformer_options"]["block_index"]
        strength = self.patch.control.get_block_strength(block_index, multiplier, weight_mask)
        args["x"] = self.patch.model_patch.model.apply(args["x"], cond_emb, block_index, "mlp_layer1", strength)
        return args


class AnimaLLLiteAdvanced(ControlBase, AdvancedControlBase):
    def __init__(self, model_patch, timestep_keyframes: TimestepKeyframeGroup):
        ControlBase.__init__(self)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controllllite())
        self.model_patch = model_patch
        self.patch = AnimaLLLiteAdvancedPatch(model_patch, self)
        self.patch_attn1 = AnimaLLLiteAdvancedAttentionPatch(
            self.patch,
            {"q": "self_attn_q_proj", "k": "self_attn_k_proj", "v": "self_attn_v_proj"},
        )
        self.patch_attn2 = AnimaLLLiteAdvancedAttentionPatch(self.patch, {"q": "cross_attn_q_proj"})
        self.patch_mlp = AnimaLLLiteAdvancedMLPPatch(self.patch)

    def prepare_batched_tensor(self, tensor: Tensor, target_batch: int, except_one=True) -> Tensor:
        if self.sub_idxs is not None:
            if tensor.shape[0] < self.full_latent_length:
                tensor = extend_to_batch_size(tensor, self.full_latent_length)
            tensor = tensor[self.sub_idxs]
        if tensor.shape[0] != target_batch:
            tensor = broadcast_image_to_extend(tensor, target_batch, self.batched_number, except_one=except_one)
        return tensor

    def prepare_effect_mask(self, mask: Tensor, img: Tensor) -> Tensor:
        mask = self.prepare_batched_tensor(mask, img.shape[0], except_one=False)
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).float(),
            size=(img.shape[2], img.shape[3]),
            mode="bilinear",
        )
        return mask.flatten(2).transpose(1, 2).to(device=img.device, dtype=img.dtype)

    def prepare_multiplier(self, img: Tensor):
        multiplier = self.strength * self._current_timestep_keyframe.strength
        token_multiplier = None

        masks = [self.mask_cond_hint_original, self._current_timestep_keyframe.mask_hint_orig]
        for mask in masks:
            if mask is not None:
                mask_multiplier = self.prepare_effect_mask(mask, img)
                token_multiplier = mask_multiplier if token_multiplier is None else token_multiplier * mask_multiplier

        flat_img = img.flatten(1, 3)
        if self.latent_keyframes is not None:
            latent_multiplier = self.calc_latent_keyframe_mults(flat_img, self.batched_number)
            token_multiplier = latent_multiplier if token_multiplier is None else token_multiplier * latent_multiplier

        if self.weights.has_uncond_multiplier and self.cond_or_uncond is not None:
            batch_multiplier = torch.ones((img.shape[0], 1, 1), dtype=img.dtype, device=img.device)
            actual_length = img.shape[0] // self.batched_number
            for idx, cond_type in enumerate(self.cond_or_uncond):
                if cond_type == 1:
                    batch_multiplier[actual_length * idx:actual_length * (idx + 1)] *= self.weights.uncond_multiplier
            token_multiplier = batch_multiplier if token_multiplier is None else token_multiplier * batch_multiplier

        weight_mask = None
        if self.weights.weight_mask is not None:
            weight_mask = self.prepare_effect_mask(self.weights.weight_mask, img)

        if token_multiplier is not None:
            multiplier = token_multiplier * multiplier
        return multiplier, weight_mask

    def get_block_strength(self, block_index: int, multiplier, weight_mask):
        block_weight = 1.0
        if self.weights.weight_type == "universal":
            exponent = self.model_patch.model.block_count - block_index
            if weight_mask is not None:
                block_weight = torch.pow(weight_mask, exponent)
            else:
                block_weight = self.weights.base_multiplier ** exponent
        elif self.weights.weights_input is not None and block_index < len(self.weights.weights_input):
            block_weight = self.weights.weights_input[block_index]
        return multiplier * block_weight

    def pre_run_advanced(self, *args, **kwargs):
        AdvancedControlBase.pre_run_advanced(self, *args, **kwargs)
        self.patch.set_control(self)

    def get_control_advanced(self, x_noisy: Tensor, t, cond, batched_number: int, transformer_options: dict):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)
        if not self.should_run():
            return control_prev

        set_model_patch(transformer_options, self.patch.set_control(self), "post_input")
        set_model_attn1_patch(transformer_options, self.patch_attn1)
        set_model_attn2_patch(transformer_options, self.patch_attn2)
        set_model_patch(transformer_options, self.patch_mlp, "mlp_patch")
        return control_prev

    def get_models(self):
        models = super().get_models()
        models.append(self.model_patch)
        return models

    def copy(self):
        copied = AnimaLLLiteAdvanced(self.model_patch, self.timestep_keyframes)
        self.copy_to(copied)
        self.copy_to_advanced(copied)
        return copied


def load_anima_lllite(ckpt_path: str, controlnet_data: dict[str, Tensor]=None, metadata=None, timestep_keyframe: TimestepKeyframeGroup=None):
    if comfy_anima_lllite is None:
        raise RuntimeError("Anima LLLite requires a newer version of ComfyUI. Please update ComfyUI.")
    if controlnet_data is None or metadata is None:
        loaded_data, loaded_metadata = comfy.utils.load_torch_file(ckpt_path, safe_load=True, return_metadata=True)
        if controlnet_data is None:
            controlnet_data = loaded_data
        metadata = loaded_metadata

    dtype = comfy.utils.weight_dtype(controlnet_data)
    model = comfy_anima_lllite.AnimaLLLite(
        controlnet_data,
        metadata,
        device=comfy.model_management.unet_offload_device(),
        dtype=dtype,
        operations=comfy.ops.manual_cast,
    )
    patcher_type = getattr(comfy.model_patcher, "CoreModelPatcher", ModelPatcher)
    model_patcher = patcher_type(
        model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )
    is_dynamic = getattr(model_patcher, "is_dynamic", lambda: False)()
    model.load_state_dict(controlnet_data, assign=is_dynamic)
    return AnimaLLLiteAdvanced(model_patcher, timestep_keyframe)


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
