#taken from: https://github.com/lllyasviel/ControlNet
#and modified
#and then taken from comfy/cldm/cldm.py and modified again

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import Tensor

from comfy.ldm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)

from comfy.cldm.cldm import ControlNet as ControlNetCLDM
from comfy.ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import PatcherInjection

from .dinklink import (InterfaceAnimateDiffInfo, InterfaceAnimateDiffModel,
                       get_CreateMotionModelPatcher, get_AnimateDiffModel, get_AnimateDiffInfo)
from .logger import logger
from .utils import (BIGMAX, AbstractPreprocWrapper, disable_weight_init_clean_groupnorm, WrapperConsts)


class SparseMotionModelPatcher(ModelPatcher):
    '''Class only used for IDE type hints.'''
    def __init__(self, *args, **kwargs):
        self.model = InterfaceAnimateDiffModel


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


def load_sparsectrl_motionmodel(ckpt_path: str, motion_data: dict[str, Tensor], ops=None) -> InterfaceAnimateDiffModel:
    mm_info: InterfaceAnimateDiffInfo = get_AnimateDiffInfo()("SD1.5", "AnimateDiff", "v3", ckpt_path)
    init_kwargs = {
        "ops": ops,
        "get_unet_func": _get_unet_func,
    }
    motion_model: InterfaceAnimateDiffModel = get_AnimateDiffModel()(mm_state_dict=motion_data, mm_info=mm_info, init_kwargs=init_kwargs)
    missing, unexpected = motion_model.load_state_dict(motion_data)
    if len(missing) > 0 or len(unexpected) > 0:
        logger.info(f"SparseCtrl MotionModel: {missing}, {unexpected}")
    return motion_model


def create_sparse_modelpatcher(model, motion_model, load_device, offload_device):
    patcher = ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    if motion_model is not None:
        _motionpatcher = _create_sparse_motionmodelpatcher(motion_model, load_device, offload_device)
        patcher.set_additional_models(WrapperConsts.ACN, [_motionpatcher])
        patcher.set_injections(WrapperConsts.ACN,
                            [PatcherInjection(inject=_inject_motion_models, eject=_eject_motion_models)])
    return patcher

def _create_sparse_motionmodelpatcher(motion_model, load_device, offload_device) -> SparseMotionModelPatcher:
    return get_CreateMotionModelPatcher()(motion_model, load_device, offload_device)


def _inject_motion_models(patcher: ModelPatcher):
    motion_models: list[SparseMotionModelPatcher] = patcher.get_additional_models_with_key(WrapperConsts.ACN)
    for mm in motion_models:
        mm.model.inject(patcher)

def _eject_motion_models(patcher: ModelPatcher):
    motion_models: list[SparseMotionModelPatcher] = patcher.get_additional_models_with_key(WrapperConsts.ACN)
    for mm in motion_models:
        mm.model.eject(patcher)

def _get_unet_func(wrapper, model: ModelPatcher):
    return model.model


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
