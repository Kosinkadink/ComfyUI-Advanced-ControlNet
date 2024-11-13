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
from .sampling import acn_sampler_sample_wrapper

DINKLINK = "__DINKLINK"

def init_dinklink():
    if not hasattr(comfy.hooks, DINKLINK):
        setattr(comfy.hooks, DINKLINK, {})
    prepare_dinklink()


def get_dinklink() -> dict[str, dict[str]]:
    return getattr(comfy.hooks, DINKLINK)

class Consts:
    ACN = "ACN"
    CREATE_SAMPLER_SAMPLE_WRAPPER = "create_sampler_sample_wrapper"

def prepare_dinklink():
    # expose acn_sampler_sample_wrapper
    d = get_dinklink()
    d.setdefault(Consts.ACN, {})[Consts.CREATE_SAMPLER_SAMPLE_WRAPPER] = None
