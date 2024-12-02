####################################################################################################
# DinkLink is my method of sharing classes/functions between my nodes.
#
# My DinkLink-compatible nodes will inject comfy.hooks with a __DINKLINK attr
# that stores a dictionary, where any of my node packs can store their stuff.
#
# It is not intended to be accessed by node packs that I don't develop, so things may change
# at any time.
#
# DinkLink also serves as a proof-of-concept for a future ComfyUI implementation of
# purposely exposing node pack classes/functions with other node packs.
####################################################################################################
from __future__ import annotations
from typing import Union
from torch import Tensor, nn

from comfy.model_patcher import ModelPatcher
import comfy.hooks

DINKLINK = "__DINKLINK"


def init_dinklink():
    create_dinklink()
    prepare_dinklink()

def create_dinklink():
    if not hasattr(comfy.hooks, DINKLINK):
        setattr(comfy.hooks, DINKLINK, {})

def get_dinklink() -> dict[str, dict[str]]:
    create_dinklink()
    return getattr(comfy.hooks, DINKLINK)


class DinkLinkConst:
    VERSION = "version"
    # ADE
    ADE = "ADE"
    ADE_ANIMATEDIFFMODEL = "AnimateDiffModel"
    ADE_ANIMATEDIFFINFO = "AnimateDiffInfo"
    ADE_CREATE_MOTIONMODELPATCHER = "create_MotionModelPatcher"

def prepare_dinklink():
    pass


class InterfaceAnimateDiffInfo:
    '''Class only used for IDE type hints; interface of ADE's AnimateDiffInfo'''
    def __init__(self, sd_type: str, mm_format: str, mm_version: str, mm_name: str):
        self.sd_type = sd_type
        self.mm_format = mm_format
        self.mm_version = mm_version
        self.mm_name = mm_name


class InterfaceAnimateDiffModel(nn.Module):
    '''Class only used for IDE type hints; interface of ADE's AnimateDiffModel'''
    def __init__(self, mm_state_dict: dict[str, Tensor], mm_info: InterfaceAnimateDiffInfo, init_kwargs: dict[str]={}):
        pass

    def set_video_length(self, video_length: int, full_length: int) -> None:
        raise NotImplemented()

    def set_scale(self, scale: Union[float, Tensor, None], per_block_list: Union[list, None]=None) -> None:
        raise NotImplemented()

    def set_effect(self, multival: Union[float, Tensor, None], per_block_list: Union[list, None]=None) -> None:
        raise NotImplemented()

    def cleanup(self):
        raise NotImplemented()

    def inject(self, model: ModelPatcher):
        pass

    def eject(self, model: ModelPatcher):
        pass


def get_CreateMotionModelPatcher(throw_exception=True):
    d = get_dinklink()
    try:
        link_ade = d[DinkLinkConst.ADE]
        return link_ade[DinkLinkConst.ADE_CREATE_MOTIONMODELPATCHER]
    except KeyError:
        if throw_exception:
            raise Exception("Could not get create_MotionModelPatcher function. AnimateDiff-Evolved nodes need to be installed to use SparseCtrl; " + \
                            "they are either not installed or are of an insufficient version.")
    return None

def get_AnimateDiffModel(throw_exception=True):
    d = get_dinklink()
    try:
        link_ade = d[DinkLinkConst.ADE]
        return link_ade[DinkLinkConst.ADE_ANIMATEDIFFMODEL]
    except KeyError:
        if throw_exception:
            raise Exception("Could not get AnimateDiffModel class. AnimateDiff-Evolved nodes need to be installed to use SparseCtrl; " + \
                            "they are either not installed or are of an insufficient version.")
    return None

def get_AnimateDiffInfo(throw_exception=True) -> InterfaceAnimateDiffInfo:
    d = get_dinklink()
    try:
        link_ade = d[DinkLinkConst.ADE]
        return link_ade[DinkLinkConst.ADE_ANIMATEDIFFINFO]
    except KeyError:
        if throw_exception:
            raise Exception("Could not get AnimateDiffInfo class - AnimateDiff-Evolved nodes need to be installed to use SparseCtrl; " + \
                            "they are either not installed or are of an insufficient version.")
    return None
