from torch import Tensor
import math

import folder_paths

from .control_plusplus import load_controlnetplusplus, PlusPlusType, PlusPlusInput, PlusPlusInputGroup, PlusPlusImageWrapper
from .utils import BIGMAX


class PlusPlusLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "plus_input": ("PLUS_INPUT", ),
                "name": (folder_paths.get_filename_list("controlnet"), ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", "IMAGE",)
    FUNCTION = "load_controlnet_plusplus"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/ControlNet++"

    def load_controlnet_plusplus(self, plus_input: PlusPlusInputGroup, name: str):
        controlnet_path = folder_paths.get_full_path("controlnet", name)
        controlnet = load_controlnetplusplus(controlnet_path)
        controlnet.verify_control_type(name, plus_input)
        controlnet.allow_condhint_latents = True
        return (controlnet, PlusPlusImageWrapper(plus_input),)


class PlusPlusLoaderSingle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": (folder_paths.get_filename_list("controlnet"), ),
                "control_type": (PlusPlusType._LIST_WITH_NONE, {"default": PlusPlusType.NONE}, ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet_plusplus"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/ControlNet++"

    def load_controlnet_plusplus(self, name: str, control_type: str):
        controlnet_path = folder_paths.get_full_path("controlnet", name)
        controlnet = load_controlnetplusplus(controlnet_path)
        controlnet.single_control_type = control_type
        controlnet.verify_control_type(name)
        return (controlnet,)


class PlusPlusInputNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "control_type": (PlusPlusType._LIST,),
            },
            "optional": {
                "prev_plus_input": ("PLUS_INPUT",),
                #"strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": BIGMAX, "step": 0.01}),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PLUS_INPUT", )
    FUNCTION = "wrap_images"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/ControlNet++"

    def wrap_images(self, image: Tensor, control_type: str, strength=1.0, prev_plus_input: PlusPlusInputGroup=None):
        if prev_plus_input is None:
            prev_plus_input = PlusPlusInputGroup()
        prev_plus_input = prev_plus_input.clone()

        if math.isclose(strength, 0.0):
            strength = 0.0000001
        pp_input = PlusPlusInput(image, control_type, strength)
        prev_plus_input.add(pp_input)

        return (prev_plus_input,)
