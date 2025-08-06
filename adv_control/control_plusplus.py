# Code ported and modified from the diffusers ControlNetPlus repo by Qi Xin:
# https://github.com/xinsir6/ControlNetPlus/blob/main/models/controlnet_union.py
from typing import Union

import os
import torch
import torch as th
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict


from comfy.ldm.modules.diffusionmodules.util import (zero_module, timestep_embedding)

from comfy.cldm.cldm import ControlNet as ControlNetCLDM
import comfy.cldm.cldm
from comfy.controlnet import ControlNet
#from comfy.t2i_adapter.adapter import ResidualAttentionBlock
from comfy.ldm.modules.attention import optimized_attention
import comfy.ops
import comfy.model_base
import comfy.model_management
import comfy.model_detection
import comfy.utils

from .utils import (AdvancedControlBase, ControlWeights, ControlWeightType, TimestepKeyframeGroup, AbstractPreprocWrapper, Extras,
                    extend_to_batch_size, broadcast_image_to_extend)
from .logger import logger


class PlusPlusType:
    OPENPOSE = "openpose"
    DEPTH = "depth"
    THICKLINE = "hed/pidi/scribble/ted"
    THINLINE = "canny/lineart/mlsd"
    NORMAL = "normal"
    SEGMENT = "segment"
    TILE = "tile"
    REPAINT = "inpaint/outpaint"
    NONE = "none"
    _LIST_WITH_NONE = [OPENPOSE, DEPTH, THICKLINE, THINLINE, NORMAL, SEGMENT, TILE, REPAINT, NONE]
    _LIST = [OPENPOSE, DEPTH, THICKLINE, THINLINE, NORMAL, SEGMENT, TILE, REPAINT]
    _DICT = {OPENPOSE: 0, DEPTH: 1, THICKLINE: 2, THINLINE: 3, NORMAL: 4, SEGMENT: 5, TILE: 6, REPAINT: 7, NONE: -1}

    @classmethod
    def to_idx(cls, control_type: str):
        try:
            return cls._DICT[control_type]
        except KeyError:
            raise Exception(f"Unknown control type '{control_type}'.")


class PlusPlusInput:
    def __init__(self, image: Tensor, control_type: str, strength: float):
        self.image = image
        self.control_type = control_type
        self.strength = strength

    def clone(self):
        return PlusPlusInput(self.image, self.control_type, self.strength)


class PlusPlusInputGroup:
    def __init__(self):
        self.controls: dict[str, PlusPlusInput] = {}
    
    def add(self, pp_input: PlusPlusInput):
        if pp_input.control_type in self.controls:
            raise Exception(f"Control type '{pp_input.control_type}' is already present; ControlNet++ does not allow more than 1 of each type.")
        self.controls[pp_input.control_type] = pp_input
    
    def clone(self) -> 'PlusPlusInputGroup':
        cloned = PlusPlusInputGroup()
        for key, value in self.controls.items():
            cloned.controls[key] = value.clone()
        return cloned


class PlusPlusImageWrapper(AbstractPreprocWrapper):
    error_msg = error_msg = "Invalid use of ControlNet++ Image Wrapper. The output of ControlNet++ Image Wrapper is NOT a usual image, but an object holding the images and extra info - you must connect the output directly to an Apply Advanced ControlNet node. It cannot be used for anything else that accepts IMAGE input."
    def __init__(self, condhint: PlusPlusInputGroup):
        super().__init__(condhint)
        # just an IDE type hint
        self.condhint: PlusPlusInputGroup

    def movedim(self, source: int, destination: int):
        condhint = self.condhint.clone()
        for pp_input in condhint.controls.values():
            pp_input.image = pp_input.image.movedim(source, destination)
        return PlusPlusImageWrapper(condhint)

# parts taken from comfy/cldm/cldm.py
class OptimizedAttention(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = nhead
        self.c = c

        self.in_proj = operations.Linear(c, c * 3, bias=True, dtype=dtype, device=device)
        self.out_proj = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.in_proj(x)
        q, k, v = x.split(self.c, dim=2)
        out = optimized_attention(q, k, v, self.heads)
        return self.out_proj(out)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResBlockUnionControlnet(nn.Module):
    def __init__(self, dim, nhead, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = OptimizedAttention(dim, nhead, dtype=dtype, device=device, operations=operations)
        self.ln_1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", operations.Linear(dim, dim * 4, dtype=dtype, device=device)), ("gelu", QuickGELU()),
                         ("c_proj", operations.Linear(dim * 4, dim, dtype=dtype, device=device))]))
        self.ln_2 = operations.LayerNorm(dim, dtype=dtype, device=device)

    def attention(self, x: torch.Tensor):
        return self.attn(x)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ControlAddEmbeddingAdv(nn.Module):
    def __init__(self, in_dim, out_dim, num_control_type, dtype=None, device=None, operations: comfy.ops.disable_weight_init=None):
        super().__init__()
        self.num_control_type = num_control_type
        self.in_dim = in_dim
        self.linear_1 = operations.Linear(in_dim * num_control_type, out_dim, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(out_dim, out_dim, dtype=dtype, device=device)

    def forward(self, control_type, dtype, device):
        if control_type is None:
            control_type = torch.zeros((self.num_control_type,), device=device)
        c_type = timestep_embedding(control_type.flatten(), self.in_dim, repeat_only=False).to(dtype).reshape((-1, self.num_control_type * self.in_dim))
        return self.linear_2(torch.nn.functional.silu(self.linear_1(c_type)))


class ControlNetPlusPlus(ControlNetCLDM):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)

        operations: comfy.ops.disable_weight_init = kwargs.get("operations", comfy.ops.disable_weight_init)
        device = kwargs.get("device", None)

        time_embed_dim = self.model_channels * 4
        control_add_embed_dim = 256

        self.control_add_embedding = ControlAddEmbeddingAdv(control_add_embed_dim, time_embed_dim, self.num_control_type, dtype=self.dtype, device=device, operations=operations)

    def union_controlnet_merge(self, hint: list[Tensor], control_type, emb, context):
        # Equivalent to: https://github.com/xinsir6/ControlNetPlus/tree/main
        indexes = torch.nonzero(control_type[0])
        inputs = []
        condition_list = []

        for idx in range(indexes.shape[0]):
            controlnet_cond = self.input_hint_block(hint[indexes[idx][0]], emb, context)
            feat_seq = torch.mean(controlnet_cond, dim=(2, 3))
            if idx < indexes.shape[0]:
                feat_seq += self.task_embedding[indexes[idx][0]].to(dtype=feat_seq.dtype, device=feat_seq.device)

            inputs.append(feat_seq.unsqueeze(1))
            condition_list.append(controlnet_cond)

        x = torch.cat(inputs, dim=1)
        x = self.transformer_layes(x)

        controlnet_cond_fuser = None
        for idx in range(indexes.shape[0]):
            alpha = self.spatial_ch_projs(x[:, idx])
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            o = condition_list[idx] + alpha
            if controlnet_cond_fuser is None:
                controlnet_cond_fuser = o
            else:
                controlnet_cond_fuser += o
        return controlnet_cond_fuser

    def forward(self, x: Tensor, hint: list[Tensor], timesteps, context, y: Tensor=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        guided_hint = None
        if self.control_add_embedding is not None:
            control_type = kwargs.get("control_type", None)

            emb += self.control_add_embedding(control_type, emb.dtype, emb.device)
            if control_type is not None:
                guided_hint = self.union_controlnet_merge(hint, control_type, emb, context)

        if guided_hint is None:
            guided_hint = self.input_hint_block(hint[0], emb, context)

        out_output = []
        out_middle = []

        hs = []
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            out_output.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        out_middle.append(self.middle_block_out(h, emb, context))

        return {"middle": out_middle, "output": out_output}


class ControlNetPlusPlusAdvanced(ControlNet, AdvancedControlBase):
    def __init__(self, control_model: ControlNetPlusPlus, timestep_keyframes: TimestepKeyframeGroup, global_average_pooling=False, load_device=None, manual_cast_dtype=None):
        super().__init__(control_model=control_model, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controlnet())
        self.add_compatible_weight(ControlWeightType.CONTROLNETPLUSPLUS)
        # for IDE type hint purposes
        self.control_model: ControlNetPlusPlus
        self.cond_hint_original: Union[PlusPlusImageWrapper, PlusPlusInputGroup]
        self.cond_hint: list[Union[Tensor, None]]
        self.cond_hint_shape: Tensor = None
        self.cond_hint_types: Tensor = None
        # in case it is using the single loader
        self.single_control_type: str = None

    def get_universal_weights(self) -> ControlWeights:
        def cn_weights_func(idx: int, control: dict[str, list[Tensor]], key: str):
            if key == "middle":
                return 1.0 * self.weights.extras.get(Extras.MIDDLE_MULT, 1.0)
            c_len = len(control[key])
            raw_weights = [(self.weights.base_multiplier ** float((c_len) - i)) for i in range(c_len+1)]
            raw_weights = raw_weights[:-1]
            if key == "input":
                raw_weights.reverse()
            return raw_weights[idx]
        return self.weights.copy_with_new_weights(new_weight_func=cn_weights_func)

    def verify_control_type(self, model_name: str, pp_group: PlusPlusInputGroup=None):
        if pp_group is not None:
            for pp_input in pp_group.controls.values():
                if PlusPlusType.to_idx(pp_input.control_type) >= self.control_model.num_control_type:
                    raise Exception(f"ControlNet++ model '{model_name}' does not support control_type '{pp_input.control_type}'.")
        if self.single_control_type is not None:
            if PlusPlusType.to_idx(self.single_control_type) >= self.control_model.num_control_type:
                raise Exception(f"ControlNet++ model '{model_name}' does not support control_type '{self.single_control_type}'.")

    def set_cond_hint_inject(self, *args, **kwargs):
        to_return = super().set_cond_hint_inject(*args, **kwargs)
        # if not single_control_type, expect PlusPlusImageWrapper
        if self.single_control_type is None:
            # check that cond_hint is wrapped, and unwrap it
            if type(self.cond_hint_original) != PlusPlusImageWrapper:
                raise Exception("ControlNet++ (Multi) expects image input from the Load ControlNet++ Model node, NOT from anything else. Images are provided to that node via ControlNet++ Input nodes.")
            self.cond_hint_original = self.cond_hint_original.condhint.clone()
        # otherwise, expect single image input (AKA, usual controlnet input)
        else:
            # check that cond_hint is not a PlusPlusImageWrapper
            if type(self.cond_hint_original) == PlusPlusImageWrapper:
                raise Exception("ControlNet++ (Single) expects usual image input, NOT the image input from a Load ControlNet++ Model (Multi) node.")
            pp_group = PlusPlusInputGroup()
            pp_input = PlusPlusInput(self.cond_hint_original, self.single_control_type, 1.0)
            pp_group.add(pp_input)
            self.cond_hint_original = pp_group
        return to_return

    def get_control_advanced(self, x_noisy: Tensor, t, cond, batched_number, transformer_options):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        output_dtype = x_noisy.dtype

        # make all cond_hints appropriate dimensions
        # TODO: change this to not require cond_hint upscaling every step when self.sub_idxs is present
        if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2] * self.compression_ratio != self.cond_hint_shape[2] or x_noisy.shape[3] * self.compression_ratio != self.cond_hint_shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = [None] * self.control_model.num_control_type
            self.cond_hint_types = torch.tensor([0.0] * self.control_model.num_control_type)
            self.cond_hint_shape = None
            compression_ratio = self.compression_ratio
            # unlike normal controlnet, need to handle each input image tensor (for each type)
            for pp_type, pp_input in self.cond_hint_original.controls.items():
                pp_idx = PlusPlusType.to_idx(pp_type)
                # if negative, means no type should be selected (single only)
                if pp_idx < 0:
                    pp_idx = 0
                else:
                    self.cond_hint_types[pp_idx] = pp_input.strength
                # if self.cond_hint_original lengths greater or equal to latent count, subdivide
                if self.sub_idxs is not None:
                    actual_cond_hint_orig = pp_input.image
                    if pp_input.image.size(0) < self.full_latent_length:
                        actual_cond_hint_orig = extend_to_batch_size(tensor=actual_cond_hint_orig, batch_size=self.full_latent_length)
                    self.cond_hint[pp_idx] = comfy.utils.common_upscale(actual_cond_hint_orig[self.sub_idxs], x_noisy.shape[3] * compression_ratio, x_noisy.shape[2] * compression_ratio, 'nearest-exact', "center")
                else:
                    self.cond_hint[pp_idx] = comfy.utils.common_upscale(pp_input.image, x_noisy.shape[3] * compression_ratio, x_noisy.shape[2] * compression_ratio, 'nearest-exact', "center")
                self.cond_hint[pp_idx] = self.cond_hint[pp_idx].to(device=x_noisy.device, dtype=dtype)
                self.cond_hint_shape = self.cond_hint[pp_idx].shape
            # prepare cond_hint_controls to match batchsize
            if self.cond_hint_types.count_nonzero() == 0:
                self.cond_hint_types = None
            else:
                self.cond_hint_types = self.cond_hint_types.unsqueeze(0).to(device=x_noisy.device, dtype=dtype).repeat(x_noisy.shape[0], 1)
        for i in range(len(self.cond_hint)):
            if self.cond_hint[i] is not None:
                if x_noisy.shape[0] != self.cond_hint[i].shape[0]:
                    self.cond_hint[i] = broadcast_image_to_extend(self.cond_hint[i], x_noisy.shape[0], batched_number)
        if self.cond_hint_types is not None and x_noisy.shape[0] != self.cond_hint_types.shape[0]:
            self.cond_hint_types = broadcast_image_to_extend(self.cond_hint_types, x_noisy.shape[0], batched_number, False)
        
        # prepare mask_cond_hint
        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number, dtype=dtype)

        context = cond.get('crossattn_controlnet', cond['c_crossattn'])
        y = cond.get('y', None)
        if y is not None:
            y = comfy.model_base.convert_tensor(y, dtype, x_noisy.device)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.float(), context=comfy.model_management.cast_to_device(context, x_noisy.device, dtype), y=y, control_type=self.cond_hint_types)
        return self.control_merge(control, control_prev, output_dtype)

    def copy(self):
        c = ControlNetPlusPlusAdvanced(self.control_model, self.timestep_keyframes, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        self.copy_to(c)
        self.copy_to_advanced(c)
        c.single_control_type = self.single_control_type
        return c


def load_controlnetplusplus(ckpt_path: str, timestep_keyframe: TimestepKeyframeGroup=None, model=None):
    controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    # check that actually is ControlNet++ model
    if "task_embedding" not in controlnet_data:
        raise Exception(f"'{ckpt_path}' is not a valid ControlNet++ model.")

    controlnet_config = None
    supported_inference_dtypes = None

    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data: #diffusers format
        controlnet_config = comfy.model_detection.unet_config_from_diffusers_unet(controlnet_data)
        diffusers_keys = comfy.utils.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                if count == 0:
                    k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                else:
                    k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                k_out = "input_hint_block.{}{}".format(count * 2, s)
                if k_in not in controlnet_data:
                    k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                    loop = False
                diffusers_keys[k_in] = k_out
            count += 1

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)
        
        if "control_add_embedding.linear_1.bias" in controlnet_data: #Union Controlnet
            controlnet_config["union_controlnet_num_control_type"] = controlnet_data["task_embedding"].shape[0]
            for k in list(controlnet_data.keys()):
                new_k = k.replace('.attn.in_proj_', '.attn.in_proj.')
                new_sd[new_k] = controlnet_data.pop(k)

        leftover_keys = controlnet_data.keys()
        if len(leftover_keys) > 0:
            logger.warning("leftover ControlNet++ keys: {}".format(leftover_keys))
        controlnet_data = new_sd
    elif "controlnet_blocks.0.weight" in controlnet_data: #SD3 diffusers format
        raise Exception("Unexpected SD3 diffusers format for ControlNet++ model. Something is very wrong.")

    pth_key = 'control_model.zero_convs.0.0.weight'
    pth = False
    key = 'zero_convs.0.0.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        raise Exception("Unexpected T2IAdapter format for ControlNet++ model. Something is very wrong.")

    if controlnet_config is None:
        model_config = comfy.model_detection.model_config_from_unet(controlnet_data, prefix, True)
        supported_inference_dtypes = model_config.supported_inference_dtypes
        controlnet_config = model_config.unet_config

    load_device = comfy.model_management.get_torch_device()
    if supported_inference_dtypes is None:
        unet_dtype = comfy.model_management.unet_dtype()
    else:
        unet_dtype = comfy.model_management.unet_dtype(supported_dtypes=supported_inference_dtypes)

    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype is not None:
        controlnet_config["operations"] = comfy.ops.manual_cast
    controlnet_config["dtype"] = unet_dtype
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
    control_model = ControlNetPlusPlus(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                comfy.model_management.load_models_gpu([model])
                model_sd = model.model_state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
            else:
                logger.warning("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)

    if len(missing) > 0:
        logger.warning("missing ControlNet++ keys: {}".format(missing))

    if len(unexpected) > 0:
        logger.debug("unexpected ControlNet++ keys: {}".format(unexpected))

    global_average_pooling = False
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"): #TODO: smarter way of enabling global_average_pooling
        global_average_pooling = True

    control = ControlNetPlusPlusAdvanced(control_model, timestep_keyframes=timestep_keyframe, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control
