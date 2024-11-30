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

def prepare_dinklink():
    pass

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

def get_AnimateDiffInfo(throw_exception=True):
    d = get_dinklink()
    try:
        link_ade = d[DinkLinkConst.ADE]
        return link_ade[DinkLinkConst.ADE_ANIMATEDIFFINFO]
    except KeyError:
        if throw_exception:
            raise Exception("Could not get AnimateDiffInfo class - AnimateDiff-Evolved nodes need to be installed to use SparseCtrl; " + \
                            "they are either not installed or are of an insufficient version.")
    return None
