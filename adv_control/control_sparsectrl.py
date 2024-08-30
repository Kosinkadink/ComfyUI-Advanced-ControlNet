#taken from: https://github.com/lllyasviel/ControlNet
#and modified
#and then taken from comfy/cldm/cldm.py and modified again

from abc import ABC, abstractmethod
import copy
import math
import numpy as np
from typing import Iterable, Union
import torch
import torch as th
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat

from comfy.ldm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)

from comfy.cli_args import args
from comfy.cldm.cldm import ControlNet as ControlNetCLDM
from comfy.ldm.modules.attention import SpatialTransformer
from comfy.ldm.modules.attention import attention_basic, attention_pytorch, attention_split, attention_sub_quad, default
from comfy.ldm.modules.attention import FeedForward, SpatialTransformer
from comfy.ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from comfy.model_patcher import ModelPatcher
import comfy.ops
import comfy.model_management
import comfy.utils

from .logger import logger
from .utils import (BIGMAX, AbstractPreprocWrapper, disable_weight_init_clean_groupnorm,
                    prepare_mask_batch, broadcast_image_to_extend, extend_to_batch_size)


# until xformers bug is fixed, do not use xformers for VersatileAttention! TODO: change this when fix is out
# logic for choosing optimized_attention method taken from comfy/ldm/modules/attention.py
# a fallback_attention_mm is selected to avoid CUDA configuration limitation with pytorch's scaled_dot_product
optimized_attention_mm = attention_basic
fallback_attention_mm = attention_basic
if comfy.model_management.xformers_enabled():
    pass
    #optimized_attention_mm = attention_xformers
if comfy.model_management.pytorch_attention_enabled():
    optimized_attention_mm = attention_pytorch
    if args.use_split_cross_attention:
        fallback_attention_mm = attention_split
    else:
        fallback_attention_mm = attention_sub_quad
else:
    if args.use_split_cross_attention:
        optimized_attention_mm = attention_split
    else:
        optimized_attention_mm = attention_sub_quad


class SparseConst:
    HINT_MULT = "sparse_hint_mult"
    NONHINT_MULT = "sparse_nonhint_mult"
    MASK_MULT = "sparse_mask_mult"


class SparseControlNet(ControlNetCLDM):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        hint_channels = kwargs.get("hint_channels")
        operations: disable_weight_init_clean_groupnorm = kwargs.get("operations", disable_weight_init_clean_groupnorm)
        device = kwargs.get("device", None)
        self.use_simplified_conditioning_embedding = kwargs.get("use_simplified_conditioning_embedding", False)
        if self.use_simplified_conditioning_embedding:
            self.input_hint_block = TimestepEmbedSequential(
                zero_module(operations.conv_nd(self.dims, hint_channels, self.model_channels, 3, padding=1, dtype=self.dtype, device=device)),
            )
        self.motion_wrapper: SparseCtrlMotionWrapper = None
    
    def set_actual_length(self, actual_length: int, full_length: int):
        if self.motion_wrapper is not None:
            self.motion_wrapper.set_video_length(video_length=actual_length, full_length=full_length)

    def forward(self, x: Tensor, hint: Tensor, timesteps, context, y=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        # SparseCtrl sets noisy input to zeros
        x = torch.zeros_like(x)
        guided_hint = self.input_hint_block(hint, emb, context)

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


class SparseModelPatcher(ModelPatcher):
    def __init__(self, *args, **kwargs):
        self.model: SparseControlNet
        super().__init__(*args, **kwargs)
    
    def load(self, device_to=None, lowvram_model_memory=0, *args, **kwargs):
        to_return = super().load(device_to=device_to, lowvram_model_memory=lowvram_model_memory, *args, **kwargs)
        if lowvram_model_memory > 0:
            self._patch_lowvram_extras(device_to=device_to)
        self._handle_float8_pe_tensors()
        return to_return

    def _patch_lowvram_extras(self, device_to=None):
        if self.model.motion_wrapper is not None:
            # figure out the tensors (likely pe's) that should be cast to device besides just the named_modules
            remaining_tensors = list(self.model.motion_wrapper.state_dict().keys())
            named_modules = []
            for n, _ in self.model.motion_wrapper.named_modules():
                named_modules.append(n)
                named_modules.append(f"{n}.weight")
                named_modules.append(f"{n}.bias")
            for name in named_modules:
                if name in remaining_tensors:
                    remaining_tensors.remove(name)

            for key in remaining_tensors:
                self.patch_weight_to_device(key, device_to)
                if device_to is not None:
                    comfy.utils.set_attr(self.model.motion_wrapper, key, comfy.utils.get_attr(self.model.motion_wrapper, key).to(device_to))

    def _handle_float8_pe_tensors(self):
        if self.model.motion_wrapper is not None:
            remaining_tensors = list(self.model.motion_wrapper.state_dict().keys())
            pe_tensors = [x for x in remaining_tensors if '.pe' in x]
            is_first = True
            for key in pe_tensors:
                if is_first:
                    is_first = False
                    if comfy.utils.get_attr(self.model.motion_wrapper, key).dtype not in [torch.float8_e5m2, torch.float8_e4m3fn]:
                        break
                comfy.utils.set_attr(self.model.motion_wrapper, key, comfy.utils.get_attr(self.model.motion_wrapper, key).half())

    # NOTE: no longer called by ComfyUI, but here for backwards compatibility
    def patch_model_lowvram(self, device_to=None, *args, **kwargs):
        patched_model = super().patch_model_lowvram(device_to, *args, **kwargs)
        self._patch_lowvram_extras(device_to=device_to)
        return patched_model

    def clone(self):
        # normal ModelPatcher clone actions
        n = SparseModelPatcher(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        if hasattr(n, "patches_uuid"):
            self.patches_uuid = n.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        if hasattr(n, "model_keys"):
            n.model_keys = self.model_keys
        if hasattr(n, "backup"):
            self.backup = n.backup
        if hasattr(n, "object_patches_backup"):
            self.object_patches_backup = n.object_patches_backup


class PreprocSparseRGBWrapper(AbstractPreprocWrapper):
    error_msg = error_msg = "Invalid use of RGB SparseCtrl output. The output of RGB SparseCtrl preprocessor is NOT a usual image, but a latent pretending to be an image - you must connect the output directly to an Apply ControlNet node (advanced or otherwise). It cannot be used for anything else that accepts IMAGE input."
    def __init__(self, condhint: Tensor):
        super().__init__(condhint)


class SparseContextAware:
    NEAREST_HINT = "nearest_hint"
    OFF = "off"

    LIST = [NEAREST_HINT, OFF]


class SparseSettings:
    def __init__(self, sparse_method: 'SparseMethod', use_motion: bool=True, motion_strength=1.0, motion_scale=1.0, merged=False,
                 sparse_mask_mult=1.0, sparse_hint_mult=1.0, sparse_nonhint_mult=1.0, context_aware=SparseContextAware.NEAREST_HINT):
        # account for Steerable-Motion workflow incompatibility;
        # doing this to for my own peace of mind (not an issue with my code)
        if type(sparse_method) == str:
            logger.warn("Outdated Steerable-Motion workflow detected; attempting to auto-convert indexes input. If you experience an error here, consult Steerable-Motion github, NOT Advanced-ControlNet.")
            sparse_method = SparseIndexMethod(get_idx_list_from_str(sparse_method))
        self.sparse_method = sparse_method
        self.use_motion = use_motion
        self.motion_strength = motion_strength
        self.motion_scale = motion_scale
        self.merged = merged
        self.sparse_mask_mult = float(sparse_mask_mult)
        self.sparse_hint_mult = float(sparse_hint_mult)
        self.sparse_nonhint_mult = float(sparse_nonhint_mult)
        self.context_aware = context_aware
    
    def is_context_aware(self):
        return self.context_aware != SparseContextAware.OFF

    @classmethod
    def default(cls):
        return SparseSettings(sparse_method=SparseSpreadMethod(), use_motion=True)


class SparseMethod(ABC):
    SPREAD = "spread"
    INDEX = "index"
    def __init__(self, method: str):
        self.method = method

    @abstractmethod
    def _get_indexes(self, hint_length: int, full_length: int) -> list[int]:
        pass

    def get_indexes(self, hint_length: int, full_length: int, sub_idxs: list[int]=None) -> tuple[list[int], list[int]]:
        returned_idxs = self._get_indexes(hint_length, full_length)
        if sub_idxs is None:
            return returned_idxs, None
        # need to map full indexes to condhint indexes
        index_mapping = {}
        for i, value in enumerate(returned_idxs):
            index_mapping[value] = i
        def get_mapped_idxs(idxs: list[int]):
            return [index_mapping[idx] for idx in idxs]
        # check if returned_idxs fit within subidxs
        fitting_idxs = []
        for sub_idx in sub_idxs:
            if sub_idx in returned_idxs:
                fitting_idxs.append(sub_idx)
        # if have any fitting_idxs, deal with it
        if len(fitting_idxs) > 0:
            return fitting_idxs, get_mapped_idxs(fitting_idxs)

        # since no returned_idxs fit in sub_idxs, need to get the next-closest hint images based on strategy
        def get_closest_idx(target_idx: int, idxs: list[int]):
            min_idx = -1
            min_dist = BIGMAX
            for idx in idxs:
                new_dist = abs(idx-target_idx)
                if new_dist < min_dist:
                    min_idx = idx
                    min_dist = new_dist
                    if min_dist == 1:
                        return min_idx, min_dist
            return min_idx, min_dist
        start_closest_idx, start_dist = get_closest_idx(sub_idxs[0], returned_idxs)
        end_closest_idx, end_dist = get_closest_idx(sub_idxs[-1], returned_idxs)
        # if only one cond hint exists, do special behavior 
        if hint_length == 1:
            # if same distance from start and end, 
            if start_dist == end_dist:
                # find center index of sub_idxs
                center_idx = sub_idxs[np.linspace(0, len(sub_idxs)-1, 3, endpoint=True, dtype=int)[1]]
                return [center_idx], get_mapped_idxs([start_closest_idx])
            # otherwise, return closest
            if start_dist < end_dist:
                return [sub_idxs[0]], get_mapped_idxs([start_closest_idx])
            return [sub_idxs[-1]], get_mapped_idxs([end_closest_idx])
        # otherwise, select up to two closest images, or just 1, whichever one applies best
        # if same distance from start and end, return two images to use 
        if start_dist == end_dist:
            return [sub_idxs[0], sub_idxs[-1]], get_mapped_idxs([start_closest_idx, end_closest_idx])
        # else, use just one
        if start_dist < end_dist:
            return [sub_idxs[0]], get_mapped_idxs([start_closest_idx])
        return [sub_idxs[-1]], get_mapped_idxs([end_closest_idx])


class SparseSpreadMethod(SparseMethod):
    UNIFORM = "uniform"
    STARTING = "starting"
    ENDING = "ending"
    CENTER = "center"

    LIST = [UNIFORM, STARTING, ENDING, CENTER]

    def __init__(self, spread=UNIFORM):
        super().__init__(self.SPREAD)
        self.spread = spread

    def _get_indexes(self, hint_length: int, full_length: int) -> list[int]:
        # if hint_length >= full_length, limit hints to full_length
        if hint_length >= full_length:
            return list(range(full_length))
        # handle special case of 1 hint image
        if hint_length == 1:
            if self.spread in [self.UNIFORM, self.STARTING]:
                return [0]
            elif self.spread == self.ENDING:
                return [full_length-1]
            elif self.spread == self.CENTER:
                # return second (of three) values as the center
                return [np.linspace(0, full_length-1, 3, endpoint=True, dtype=int)[1]]
            else:
                raise ValueError(f"Unrecognized spread: {self.spread}")
        # otherwise, handle other cases
        if self.spread == self.UNIFORM:
            return list(np.linspace(0, full_length-1, hint_length, endpoint=True, dtype=int))
        elif self.spread == self.STARTING:
            # make split 1 larger, remove last element
            return list(np.linspace(0, full_length-1, hint_length+1, endpoint=True, dtype=int))[:-1]
        elif self.spread == self.ENDING:
            # make split 1 larger, remove first element
            return list(np.linspace(0, full_length-1, hint_length+1, endpoint=True, dtype=int))[1:]
        elif self.spread == self.CENTER:
            # if hint length is not 3 greater than full length, do STARTING behavior
            if full_length-hint_length < 3:
                return list(np.linspace(0, full_length-1, hint_length+1, endpoint=True, dtype=int))[:-1]
            # otherwise, get linspace of 2 greater than needed, then cut off first and last
            return list(np.linspace(0, full_length-1, hint_length+2, endpoint=True, dtype=int))[1:-1]
        return ValueError(f"Unrecognized spread: {self.spread}")


class SparseIndexMethod(SparseMethod):
    def __init__(self, idxs: list[int]):
        super().__init__(self.INDEX)
        self.idxs = idxs

    def _get_indexes(self, hint_length: int, full_length: int) -> list[int]:
        orig_hint_length = hint_length
        if hint_length > full_length:
            hint_length = full_length
        # if idxs is less than hint_length, throw error
        if len(self.idxs) < hint_length:
            err_msg = f"There are not enough indexes ({len(self.idxs)}) provided to fit the usable {hint_length} input images."
            if orig_hint_length != hint_length:
                err_msg = f"{err_msg} (original input images: {orig_hint_length})"
            raise ValueError(err_msg)
        # cap idxs to hint_length
        idxs = self.idxs[:hint_length]
        new_idxs = []
        real_idxs = set()
        for idx in idxs:
            if idx < 0:
                real_idx = full_length+idx
                if real_idx in real_idxs:
                    raise ValueError(f"Index '{idx}' maps to '{real_idx}' and is duplicate - indexes in Sparse Index Method must be unique.")
            else:
                real_idx = idx
                if real_idx in real_idxs:
                    raise ValueError(f"Index '{idx}' is duplicate (or a negative index is equivalent) - indexes in Sparse Index Method must be unique.")
            real_idxs.add(real_idx)
            new_idxs.append(real_idx)
        return new_idxs  


def get_idx_list_from_str(indexes: str) -> list[int]:
    idxs = []
    unique_idxs = set()
    # get indeces from string
    str_idxs = [x.strip() for x in indexes.strip().split(",")]
    for str_idx in str_idxs:
        try:
            idx = int(str_idx)
            if idx in unique_idxs:
                raise ValueError(f"'{idx}' is duplicated; indexes must be unique.")
            idxs.append(idx)
            unique_idxs.add(idx)
        except ValueError:
            raise ValueError(f"'{str_idx}' is not a valid integer index.")
    if len(idxs) == 0:
        raise ValueError(f"No indexes were listed in Sparse Index Method.")
    return idxs


#########################################
# motion-related portion of controlnet
class BlockType:
    UP = "up"
    DOWN = "down"
    MID = "mid"

def get_down_block_max(mm_state_dict: dict[str, Tensor]) -> int:
    return get_block_max(mm_state_dict, "down_blocks")

def get_up_block_max(mm_state_dict: dict[str, Tensor]) -> int:
    return get_block_max(mm_state_dict, "up_blocks")

def get_block_max(mm_state_dict: dict[str, Tensor], block_name: str) -> int:
    # keep track of biggest down_block count in module
    biggest_block = -1
    for key in mm_state_dict.keys():
        if block_name in key:
            try:
                block_int = key.split(".")[1]
                block_num = int(block_int)
                if block_num > biggest_block:
                    biggest_block = block_num
            except ValueError:
                pass
    return biggest_block

def has_mid_block(mm_state_dict: dict[str, Tensor]):
    # check if keys contain mid_block
    for key in mm_state_dict.keys():
        if key.startswith("mid_block."):
            return True
    return False

def get_position_encoding_max_len(mm_state_dict: dict[str, Tensor], mm_name: str=None) -> int:
    # use pos_encoder.pe entries to determine max length - [1, {max_length}, {320|640|1280}]
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.pe"):
            return mm_state_dict[key].size(1) # get middle dim
    raise ValueError(f"No pos_encoder.pe found in SparseCtrl state_dict - {mm_name} is not a valid SparseCtrl model!")


class SparseCtrlMotionWrapper(nn.Module):
    def __init__(self, mm_state_dict: dict[str, Tensor], ops=disable_weight_init_clean_groupnorm):
        super().__init__()
        self.down_blocks: Iterable[MotionModule] = None
        self.up_blocks: Iterable[MotionModule] = None
        self.mid_block: MotionModule = None
        self.encoding_max_len = get_position_encoding_max_len(mm_state_dict, "")
        layer_channels = (320, 640, 1280, 1280)
        if get_down_block_max(mm_state_dict) > -1:
            self.down_blocks = nn.ModuleList([])
            for c in layer_channels:
                self.down_blocks.append(MotionModule(c, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.DOWN, ops=ops))
        if get_up_block_max(mm_state_dict) > -1:
            self.up_blocks = nn.ModuleList([])
            for c in reversed(layer_channels):
                self.up_blocks.append(MotionModule(c, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.UP, ops=ops))
        if has_mid_block(mm_state_dict):
            self.mid_block = MotionModule(1280, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.MID, ops=ops)

    def inject(self, unet: SparseControlNet):
        # inject input (down) blocks
        self._inject(unet.input_blocks, self.down_blocks)
        # inject mid block, if present
        if self.mid_block is not None:
            self._inject([unet.middle_block], [self.mid_block])
        unet.motion_wrapper = self

    def _inject(self, unet_blocks: nn.ModuleList, mm_blocks: nn.ModuleList):
        # Rules for injection:
        # For each component list in a unet block:
        #     if SpatialTransformer exists in list, place next block after last occurrence
        #     elif ResBlock exists in list, place next block after first occurrence
        #     else don't place block
        injection_count = 0
        unet_idx = 0
        # details about blocks passed in
        per_block = len(mm_blocks[0].motion_modules)
        injection_goal = len(mm_blocks) * per_block
        # only stop injecting when modules exhausted
        while injection_count < injection_goal:
            # figure out which VanillaTemporalModule from mm to inject
            mm_blk_idx, mm_vtm_idx = injection_count // per_block, injection_count % per_block
            # figure out layout of unet block components
            st_idx = -1 # SpatialTransformer index
            res_idx = -1 # first ResBlock index
            # first, figure out indeces of relevant blocks
            for idx, component in enumerate(unet_blocks[unet_idx]):
                if type(component) == SpatialTransformer:
                    st_idx = idx
                elif type(component).__name__ == "ResBlock" and res_idx < 0:
                    res_idx = idx
            # if SpatialTransformer exists, inject right after
            if st_idx >= 0:
                unet_blocks[unet_idx].insert(st_idx+1, mm_blocks[mm_blk_idx].motion_modules[mm_vtm_idx])
                injection_count += 1
            # otherwise, if only ResBlock exists, inject right after
            elif res_idx >= 0:
                unet_blocks[unet_idx].insert(res_idx+1, mm_blocks[mm_blk_idx].motion_modules[mm_vtm_idx])
                injection_count += 1
            # increment unet_idx
            unet_idx += 1

    def eject(self, unet: SparseControlNet):
        # remove from input blocks (downblocks)
        self._eject(unet.input_blocks)
        # remove from middle block (encapsulate in list to make compatible)
        self._eject([unet.middle_block])
        del unet.motion_wrapper
        unet.motion_wrapper = None

    def _eject(self, unet_blocks: nn.ModuleList):
        # eject all VanillaTemporalModule objects from all blocks
        for block in unet_blocks:
            idx_to_pop = []
            for idx, component in enumerate(block):
                if type(component) == VanillaTemporalModule:
                    idx_to_pop.append(idx)
            # pop in backwards order, as to not disturb what the indeces refer to
            for idx in sorted(idx_to_pop, reverse=True):
                block.pop(idx)

    def set_video_length(self, video_length: int, full_length: int):
        self.AD_video_length = video_length
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_video_length(video_length, full_length)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_video_length(video_length, full_length)
        if self.mid_block is not None:
            self.mid_block.set_video_length(video_length, full_length)
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_scale_multiplier(multiplier)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_scale_multiplier(multiplier)
        if self.mid_block is not None:
            self.mid_block.set_scale_multiplier(multiplier)

    def set_strength(self, strength: float):
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_strength(strength)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_strength(strength)
        if self.mid_block is not None:
            self.mid_block.set_strength(strength)

    def reset_temp_vars(self):
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.reset_temp_vars()
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.reset_temp_vars()
        if self.mid_block is not None:
            self.mid_block.reset_temp_vars()

    def reset_scale_multiplier(self):
        self.set_scale_multiplier(None)

    def reset(self):
        self.reset_scale_multiplier()
        self.reset_temp_vars()


class MotionModule(nn.Module):
    def __init__(self, in_channels, temporal_position_encoding_max_len=24, block_type: str=BlockType.DOWN, ops=disable_weight_init_clean_groupnorm):
        super().__init__()
        if block_type == BlockType.MID:
            # mid blocks contain only a single VanillaTemporalModule
            self.motion_modules: Iterable[VanillaTemporalModule] = nn.ModuleList([get_motion_module(in_channels, temporal_position_encoding_max_len, ops=ops)])
        else:
            # down blocks contain two VanillaTemporalModules
            self.motion_modules: Iterable[VanillaTemporalModule] = nn.ModuleList(
                [
                    get_motion_module(in_channels, temporal_position_encoding_max_len, ops=ops),
                    get_motion_module(in_channels, temporal_position_encoding_max_len, ops=ops)
                ]
            )
            # up blocks contain one additional VanillaTemporalModule
            if block_type == BlockType.UP:
                self.motion_modules.append(get_motion_module(in_channels, temporal_position_encoding_max_len, ops=ops))
    
    def set_video_length(self, video_length: int, full_length: int):
        for motion_module in self.motion_modules:
            motion_module.set_video_length(video_length, full_length)
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for motion_module in self.motion_modules:
            motion_module.set_scale_multiplier(multiplier)
    
    def set_masks(self, masks: Tensor, min_val: float, max_val: float):
        for motion_module in self.motion_modules:
            motion_module.set_masks(masks, min_val, max_val)
    
    def set_sub_idxs(self, sub_idxs: list[int]):
        for motion_module in self.motion_modules:
            motion_module.set_sub_idxs(sub_idxs)

    def set_strength(self, strength: float):
        for motion_module in self.motion_modules:
            motion_module.set_strength(strength)

    def reset_temp_vars(self):
        for motion_module in self.motion_modules:
            motion_module.reset_temp_vars()


def get_motion_module(in_channels, temporal_position_encoding_max_len, ops=disable_weight_init_clean_groupnorm):
    # unlike normal AD, there is only one attention block expected in SparseCtrl models
    return VanillaTemporalModule(in_channels=in_channels, attention_block_types=("Temporal_Self",), temporal_position_encoding_max_len=temporal_position_encoding_max_len, ops=ops)


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=1,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=True,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
        ops=disable_weight_init_clean_groupnorm,
    ):
        super().__init__()
        self.strength = 1.0
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            ops=ops,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def set_video_length(self, video_length: int, full_length: int):
        self.temporal_transformer.set_video_length(video_length, full_length)
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        self.temporal_transformer.set_scale_multiplier(multiplier)

    def set_masks(self, masks: Tensor, min_val: float, max_val: float):
        self.temporal_transformer.set_masks(masks, min_val, max_val)
    
    def set_sub_idxs(self, sub_idxs: list[int]):
        self.temporal_transformer.set_sub_idxs(sub_idxs)

    def set_strength(self, strength: float):
        self.strength = strength

    def reset_temp_vars(self):
        self.set_strength(1.0)
        self.temporal_transformer.reset_temp_vars()

    def forward(self, input_tensor, encoder_hidden_states=None, attention_mask=None):
        if math.isclose(self.strength, 1.0):
            return self.temporal_transformer(input_tensor, encoder_hidden_states, attention_mask)
        elif math.isclose(self.strength, 0.0):
            return input_tensor
        # elif self.strength > 1.0:
        #     return self.temporal_transformer(input_tensor, encoder_hidden_states, attention_mask)*self.strength
        else:
            return self.temporal_transformer(input_tensor, encoder_hidden_states, attention_mask)*self.strength + input_tensor*(1.0-self.strength)


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        ops=disable_weight_init_clean_groupnorm,
    ):
        super().__init__()
        self.video_length = 16
        self.full_length = 16
        self.scale_min = 1.0
        self.scale_max = 1.0
        self.raw_scale_mask: Union[Tensor, None] = None
        self.temp_scale_mask: Union[Tensor, None] = None
        self.sub_idxs: Union[list[int], None] = None
        self.prev_hidden_states_batch = 0


        inner_dim = num_attention_heads * attention_head_dim

        self.norm = ops.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = ops.Linear(in_channels, inner_dim)

        self.transformer_blocks: Iterable[TemporalTransformerBlock] = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    ops=ops,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = ops.Linear(inner_dim, in_channels)

    def set_video_length(self, video_length: int, full_length: int):
        self.video_length = video_length
        self.full_length = full_length
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for block in self.transformer_blocks:
            block.set_scale_multiplier(multiplier)

    def set_masks(self, masks: Tensor, min_val: float, max_val: float):
        self.scale_min = min_val
        self.scale_max = max_val
        self.raw_scale_mask = masks

    def set_sub_idxs(self, sub_idxs: list[int]):
        self.sub_idxs = sub_idxs
        for block in self.transformer_blocks:
            block.set_sub_idxs(sub_idxs)

    def reset_temp_vars(self):
        del self.temp_scale_mask
        self.temp_scale_mask = None
        self.prev_hidden_states_batch = 0
        for block in self.transformer_blocks:
            block.reset_temp_vars()

    def get_scale_mask(self, hidden_states: Tensor) -> Union[Tensor, None]:
        # if no raw mask, return None
        if self.raw_scale_mask is None:
            return None
        shape = hidden_states.shape
        batch, channel, height, width = shape
        # if temp mask already calculated, return it
        if self.temp_scale_mask != None:
            # check if hidden_states batch matches
            if batch == self.prev_hidden_states_batch:
                if self.sub_idxs is not None:
                    return self.temp_scale_mask[:, self.sub_idxs, :]
                return self.temp_scale_mask
            # if does not match, reset cached temp_scale_mask and recalculate it
            del self.temp_scale_mask
            self.temp_scale_mask = None
        # otherwise, calculate temp mask
        self.prev_hidden_states_batch = batch
        mask = prepare_mask_batch(self.raw_scale_mask, shape=(self.full_length, 1, height, width))
        mask = extend_to_batch_size(mask, self.full_length)
        # if mask not the same amount length as full length, make it match
        if self.full_length != mask.shape[0]:
            mask = broadcast_image_to_extend(mask, self.full_length, 1)
        # reshape mask to attention K shape (h*w, latent_count, 1)
        batch, channel, height, width = mask.shape
        # first, perform same operations as on hidden_states,
        # turning (b, c, h, w) -> (b, h*w, c)
        mask = mask.permute(0, 2, 3, 1).reshape(batch, height*width, channel)
        # then, make it the same shape as attention's k, (h*w, b, c)
        mask = mask.permute(1, 0, 2)
        # make masks match the expected length of h*w
        batched_number = shape[0] // self.video_length
        if batched_number > 1:
            mask = torch.cat([mask] * batched_number, dim=0)
        # cache mask and set to proper device
        self.temp_scale_mask = mask
        # move temp_scale_mask to proper dtype + device
        self.temp_scale_mask = self.temp_scale_mask.to(dtype=hidden_states.dtype, device=hidden_states.device)
        # return subset of masks, if needed
        if self.sub_idxs is not None:
            return self.temp_scale_mask[:, self.sub_idxs, :]
        return self.temp_scale_mask

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states
        scale_mask = self.get_scale_mask(hidden_states)
        # add some casts for fp8 purposes - does not affect speed otherwise
        hidden_states = self.norm(hidden_states).to(hidden_states.dtype)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states).to(hidden_states.dtype)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                video_length=self.video_length,
                scale_mask=scale_mask
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        output = hidden_states + residual

        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        ops=disable_weight_init_clean_groupnorm,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    context_dim=cross_attention_dim # called context_dim for ComfyUI impl
                    if block_name.endswith("_Cross")
                    else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    #bias=attention_bias, # remove for Comfy CrossAttention
                    #upcast_attention=upcast_attention, # remove for Comfy CrossAttention
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    ops=ops,
                )
            )
            norms.append(ops.LayerNorm(dim))

        self.attention_blocks: Iterable[VersatileAttention] = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=(activation_fn == "geglu"), operations=ops)
        self.ff_norm = ops.LayerNorm(dim)

    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for block in self.attention_blocks:
            block.set_scale_multiplier(multiplier)

    def set_sub_idxs(self, sub_idxs: list[int]):
        for block in self.attention_blocks:
            block.set_sub_idxs(sub_idxs)

    def reset_temp_vars(self):
        for block in self.attention_blocks:
            block.reset_temp_vars()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        scale_mask=None
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states).to(hidden_states.dtype)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if attention_block.is_cross_attention
                    else None,
                    attention_mask=attention_mask,
                    video_length=video_length,
                    scale_mask=scale_mask
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.sub_idxs = None

    def set_sub_idxs(self, sub_idxs: list[int]):
        self.sub_idxs = sub_idxs

    def forward(self, x):
        #if self.sub_idxs is not None:
        #    x = x + self.pe[:, self.sub_idxs]
        #else:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CrossAttentionMMSparse(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, device=None,
                 operations=disable_weight_init_clean_groupnorm):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.actual_attention = optimized_attention_mm
        self.heads = heads
        self.dim_head = dim_head
        self.scale = None

        self.to_q = operations.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

    def reset_attention_type(self):
        self.actual_attention = optimized_attention_mm

    def forward(self, x, context=None, value=None, mask=None, scale_mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k: Tensor = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        # apply custom scale by multiplying k by scale factor
        if self.scale is not None:
            k *= self.scale
        
        # apply scale mask, if present
        if scale_mask is not None:
            k *= scale_mask

        try:
            out = self.actual_attention(q, k, v, self.heads, mask)
        except RuntimeError as e:
            if str(e).startswith("CUDA error: invalid configuration argument"):
                self.actual_attention = fallback_attention_mm
                out = self.actual_attention(q, k, v, self.heads, mask)
            else:
                raise
        return self.to_out(out)


class VersatileAttention(CrossAttentionMMSparse):
    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        ops=disable_weight_init_clean_groupnorm,
        *args,
        **kwargs,
    ):
        super().__init__(operations=ops, *args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["context_dim"] is not None

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def set_scale_multiplier(self, multiplier: Union[float, None]):
        if multiplier is None or math.isclose(multiplier, 1.0):
            self.scale = None
        else:
            self.scale = multiplier

    def set_sub_idxs(self, sub_idxs: list[int]):
        if self.pos_encoder != None:
            self.pos_encoder.set_sub_idxs(sub_idxs)

    def reset_temp_vars(self):
        self.reset_attention_type()

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        scale_mask=None,
    ):
        if self.attention_mode != "Temporal":
            raise NotImplementedError

        d = hidden_states.shape[1]
        hidden_states = rearrange(
            hidden_states, "(b f) d c -> (b d) f c", f=video_length
        )

        if self.pos_encoder is not None:
           hidden_states = self.pos_encoder(hidden_states).to(hidden_states.dtype)

        encoder_hidden_states = (
            repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
            if encoder_hidden_states is not None
            else encoder_hidden_states
        )

        hidden_states = super().forward(
            hidden_states,
            encoder_hidden_states,
            value=None,
            mask=attention_mask,
            scale_mask=scale_mask,
        )

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
