# adapted from https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI
# basically, all the LLLite core code is from there, which I then combined with
# Advanced-ControlNet features and QoL
import math
from typing import Union
from torch import Tensor
import torch
import os

import comfy.utils
from comfy.controlnet import ControlBase

from .logger import logger
from .utils import AdvancedControlBase, prepare_mask_batch


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
    def __init__(self, modules: dict[str, 'LLLiteModule'], control: Union[AdvancedControlBase, ControlBase]=None):
        self.modules = modules
        self.control = control
    
    def __call__(self, q, k, v, extra_options):
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
        for d in self.modules.keys():
            self.modules[d] = self.modules[d].to(device)
        return self
    
    def set_control(self, control: Union[AdvancedControlBase, ControlBase]):
        self.control = control

    def clone_with_control(self, control: AdvancedControlBase):
        return LLLitePatch(self.modules, control)

    def cleanup(self):
        del self.control
        self.control = None
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
        self.cond_emb = None
        self.cx_shape = None
        self.prev_batch = 0
        self.prev_sub_idxs = None

    def forward(self, x: Tensor, control: Union[AdvancedControlBase, ControlBase]):
        mask = None
        mask_tk = None
        if self.cond_emb is None or control.sub_idxs != self.prev_sub_idxs or x.shape[0] != self.prev_batch:
            # print(f"cond_emb is None, {self.name}")
            cx = self.conditioning1(control.cond_hint.to(x.device, dtype=x.dtype))
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

        cx = torch.cat([cx, self.down(x)], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)
        cx = self.up(cx)
        if control.latent_keyframes is not None:
            cx = cx * control.calc_latent_keyframe_mults(x=cx, batched_number=control.batched_number)
        return cx * mask * control.strength * control.current_timestep_keyframe.strength
