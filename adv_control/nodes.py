import numpy as np
from torch import Tensor

import folder_paths
from comfy.model_patcher import ModelPatcher

from .control import load_controlnet, convert_to_advanced, is_advanced_controlnet
from .utils import ControlWeights, LatentKeyframeGroup, TimestepKeyframeGroup, BIGMAX
from .nodes_weight import (DefaultWeights, ScaledSoftMaskedUniversalWeights, ScaledSoftUniversalWeights, SoftControlNetWeights, CustomControlNetWeights,
    SoftT2IAdapterWeights, CustomT2IAdapterWeights)
from .nodes_keyframes import (LatentKeyframeGroupNode, LatentKeyframeInterpolationNode, LatentKeyframeBatchedGroupNode, LatentKeyframeNode,
                              TimestepKeyframeNode, TimestepKeyframeInterpolationNode, TimestepKeyframeFromStrengthListNode)
from .nodes_sparsectrl import SparseCtrlMergedLoaderAdvanced, SparseCtrlLoaderAdvanced, SparseIndexMethodNode, SparseSpreadMethodNode, RgbSparseCtrlPreprocessor
from .nodes_reference import ReferenceControlNetNode, ReferenceControlFinetune, ReferencePreprocessorNode
from .nodes_loosecontrol import ControlNetLoaderWithLoraAdvanced
from .nodes_deprecated import LoadImagesFromDirectory
from .logger import logger


class ControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            },
            "optional": {
                "timestep_keyframe": ("TIMESTEP_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def load_controlnet(self, control_net_name,
                        timestep_keyframe: TimestepKeyframeGroup=None
                        ):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, timestep_keyframe)
        return (controlnet,)
    

class DiffControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), )
            },
            "optional": {
                "timestep_keyframe": ("TIMESTEP_KEYFRAME", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def load_controlnet(self, control_net_name, model,
                        timestep_keyframe: TimestepKeyframeGroup=None
                        ):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, timestep_keyframe, model)
        if is_advanced_controlnet(controlnet):
            controlnet.verify_all_weights()
        return (controlnet,)


class AdvancedControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "mask_optional": ("MASK", ),
                "timestep_kf": ("TIMESTEP_KEYFRAME", ),
                "latent_kf_override": ("LATENT_KEYFRAME", ),
                "weights_override": ("CONTROL_NET_WEIGHTS", ),
                "model_optional": ("MODEL",),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","MODEL",)
    RETURN_NAMES = ("positive", "negative", "model_opt")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None, model_optional: ModelPatcher=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None):
        if strength == 0:
            return (positive, negative, model_optional)
        if model_optional:
            model_optional = model_optional.clone()

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    # copy, convert to advanced if needed, and set cond
                    c_net = convert_to_advanced(control_net.copy()).set_cond_hint(control_hint, strength, (start_percent, end_percent))
                    if is_advanced_controlnet(c_net):
                        # disarm node check
                        c_net.disarm()
                        # if model required, verify model is passed in, and if so patch it
                        if c_net.require_model:
                            if not model_optional:
                                raise Exception(f"Type '{type(c_net).__name__}' requires model_optional input, but got None.")
                            c_net.patch_model(model=model_optional)
                        # apply optional parameters and overrides, if provided
                        if timestep_kf is not None:
                            c_net.set_timestep_keyframes(timestep_kf)
                        if latent_kf_override is not None:
                            c_net.latent_keyframe_override = latent_kf_override
                        if weights_override is not None:
                            c_net.weights_override = weights_override
                        # verify weights are compatible
                        c_net.verify_all_weights()
                        # set cond hint mask
                        if mask_optional is not None:
                            mask_optional = mask_optional.clone()
                            # if not in the form of a batch, make it so
                            if len(mask_optional.shape) < 3:
                                mask_optional = mask_optional.unsqueeze(0)
                            c_net.set_cond_hint_mask(mask_optional)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], model_optional)


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": TimestepKeyframeNode,
    "ACN_TimestepKeyframeInterpolation": TimestepKeyframeInterpolationNode,
    "ACN_TimestepKeyframeFromStrengthList": TimestepKeyframeFromStrengthListNode,
    "LatentKeyframe": LatentKeyframeNode,
    "LatentKeyframeTiming": LatentKeyframeInterpolationNode,
    "LatentKeyframeBatchedGroup": LatentKeyframeBatchedGroupNode,
    "LatentKeyframeGroup": LatentKeyframeGroupNode,
    # Conditioning
    "ACN_AdvancedControlNetApply": AdvancedControlNetApply,
    # Loaders
    "ControlNetLoaderAdvanced": ControlNetLoaderAdvanced,
    "DiffControlNetLoaderAdvanced": DiffControlNetLoaderAdvanced,
    # Weights
    "ScaledSoftControlNetWeights": ScaledSoftUniversalWeights,
    "ScaledSoftMaskedUniversalWeights": ScaledSoftMaskedUniversalWeights,
    "SoftControlNetWeights": SoftControlNetWeights,
    "CustomControlNetWeights": CustomControlNetWeights,
    "SoftT2IAdapterWeights": SoftT2IAdapterWeights,
    "CustomT2IAdapterWeights": CustomT2IAdapterWeights,
    "ACN_DefaultUniversalWeights": DefaultWeights,
    # SparseCtrl
    "ACN_SparseCtrlRGBPreprocessor": RgbSparseCtrlPreprocessor,
    "ACN_SparseCtrlLoaderAdvanced": SparseCtrlLoaderAdvanced,
    "ACN_SparseCtrlMergedLoaderAdvanced": SparseCtrlMergedLoaderAdvanced,
    "ACN_SparseCtrlIndexMethodNode": SparseIndexMethodNode,
    "ACN_SparseCtrlSpreadMethodNode": SparseSpreadMethodNode,
    # Reference
    "ACN_ReferencePreprocessor": ReferencePreprocessorNode,
    "ACN_ReferenceControlNet": ReferenceControlNetNode,
    "ACN_ReferenceControlNetFinetune": ReferenceControlFinetune,
    # LOOSEControl
    #"ACN_ControlNetLoaderWithLoraAdvanced": ControlNetLoaderWithLoraAdvanced,
    # Deprecated
    "LoadImagesFromDirectory": LoadImagesFromDirectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": "Timestep Keyframe 🛂🅐🅒🅝",
    "ACN_TimestepKeyframeInterpolation": "Timestep Keyframe Interpolation 🛂🅐🅒🅝",
    "ACN_TimestepKeyframeFromStrengthList": "Timestep Keyframe From List 🛂🅐🅒🅝",
    "LatentKeyframe": "Latent Keyframe 🛂🅐🅒🅝",
    "LatentKeyframeTiming": "Latent Keyframe Interpolation 🛂🅐🅒🅝",
    "LatentKeyframeBatchedGroup": "Latent Keyframe From List 🛂🅐🅒🅝",
    "LatentKeyframeGroup": "Latent Keyframe Group 🛂🅐🅒🅝",
    # Conditioning
    "ACN_AdvancedControlNetApply": "Apply Advanced ControlNet 🛂🅐🅒🅝",
    # Loaders
    "ControlNetLoaderAdvanced": "Load Advanced ControlNet Model 🛂🅐🅒🅝",
    "DiffControlNetLoaderAdvanced": "Load Advanced ControlNet Model (diff) 🛂🅐🅒🅝",
    # Weights
    "ScaledSoftControlNetWeights": "Scaled Soft Weights 🛂🅐🅒🅝",
    "ScaledSoftMaskedUniversalWeights": "Scaled Soft Masked Weights 🛂🅐🅒🅝",
    "SoftControlNetWeights": "ControlNet Soft Weights 🛂🅐🅒🅝",
    "CustomControlNetWeights": "ControlNet Custom Weights 🛂🅐🅒🅝",
    "SoftT2IAdapterWeights": "T2IAdapter Soft Weights 🛂🅐🅒🅝",
    "CustomT2IAdapterWeights": "T2IAdapter Custom Weights 🛂🅐🅒🅝",
    "ACN_DefaultUniversalWeights": "Force Default Weights 🛂🅐🅒🅝",
    # SparseCtrl
    "ACN_SparseCtrlRGBPreprocessor": "RGB SparseCtrl 🛂🅐🅒🅝",
    "ACN_SparseCtrlLoaderAdvanced": "Load SparseCtrl Model 🛂🅐🅒🅝",
    "ACN_SparseCtrlMergedLoaderAdvanced": "🧪Load Merged SparseCtrl Model 🛂🅐🅒🅝",
    "ACN_SparseCtrlIndexMethodNode": "SparseCtrl Index Method 🛂🅐🅒🅝",
    "ACN_SparseCtrlSpreadMethodNode": "SparseCtrl Spread Method 🛂🅐🅒🅝",
    # Reference
    "ACN_ReferencePreprocessor": "Reference Preproccessor 🛂🅐🅒🅝",
    "ACN_ReferenceControlNet": "Reference ControlNet 🛂🅐🅒🅝",
    "ACN_ReferenceControlNetFinetune": "Reference ControlNet (Finetune) 🛂🅐🅒🅝",
    # LOOSEControl
    #"ACN_ControlNetLoaderWithLoraAdvanced": "Load Adv. ControlNet Model w/ LoRA 🛂🅐🅒🅝",
    # Deprecated
    "LoadImagesFromDirectory": "🚫Load Images [DEPRECATED] 🛂🅐🅒🅝",
}
