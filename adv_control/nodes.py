import numpy as np
from torch import Tensor

import folder_paths

from .control import load_controlnet, convert_to_advanced, is_advanced_controlnet
from .utils import ControlWeights, ControlWeightType, LatentKeyframeGroup, TimestepKeyframe, TimestepKeyframeGroup
from .utils import StrengthInterpolation as SI
from .nodes_weight import (DefaultWeights, ScaledSoftMaskedUniversalWeights, ScaledSoftUniversalWeights, SoftControlNetWeights, CustomControlNetWeights,
    SoftT2IAdapterWeights, CustomT2IAdapterWeights)
from .nodes_latent_keyframe import LatentKeyframeGroupNode, LatentKeyframeInterpolationNode, LatentKeyframeBatchedGroupNode, LatentKeyframeNode
from .nodes_sparsectrl import SparseCtrlMergedLoaderAdvanced, SparseCtrlLoaderAdvanced, SparseIndexMethodNode, SparseSpreadMethodNode, RgbSparseCtrlPreprocessor
from .nodes_loosecontrol import ControlNetLoaderWithLoraAdvanced
from .nodes_deprecated import LoadImagesFromDirectory
from .logger import logger


class TimestepKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
            },
            "optional": {
                "prev_timestep_kf": ("TIMESTEP_KEYFRAME", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "cn_weights": ("CONTROL_NET_WEIGHTS", ),
                "latent_keyframe": ("LATENT_KEYFRAME", ),
                "null_latent_kf_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "guarantee_usage": ("BOOLEAN", {"default": True}, ),
                "mask_optional": ("MASK", ),
                #"interpolation": ([SI.LINEAR, SI.EASE_IN, SI.EASE_OUT, SI.EASE_IN_OUT, SI.NONE], {"default": SI.NONE}, ),
            }
        }
    
    RETURN_NAMES = ("TIMESTEP_KF", )
    RETURN_TYPES = ("TIMESTEP_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                      start_percent: float,
                      strength: float=1.0,
                      cn_weights: ControlWeights=None, control_net_weights: ControlWeights=None, # old name
                      latent_keyframe: LatentKeyframeGroup=None,
                      prev_timestep_kf: TimestepKeyframeGroup=None, prev_timestep_keyframe: TimestepKeyframeGroup=None, # old name
                      null_latent_kf_strength: float=0.0,
                      inherit_missing=True,
                      guarantee_usage=True,
                      mask_optional=None,
                      interpolation: str=SI.NONE,):
        control_net_weights = control_net_weights if control_net_weights else cn_weights
        prev_timestep_keyframe = prev_timestep_keyframe if prev_timestep_keyframe else prev_timestep_kf
        if not prev_timestep_keyframe:
            prev_timestep_keyframe = TimestepKeyframeGroup()
        else:
            prev_timestep_keyframe = prev_timestep_keyframe.clone()
        keyframe = TimestepKeyframe(start_percent=start_percent, strength=strength, interpolation=interpolation, null_latent_kf_strength=null_latent_kf_strength,
                                    control_weights=control_net_weights, latent_keyframes=latent_keyframe, inherit_missing=inherit_missing, guarantee_usage=guarantee_usage,
                                    mask_hint_orig=mask_optional)
        prev_timestep_keyframe.add(keyframe)
        return (prev_timestep_keyframe,)


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
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None):
        if strength == 0:
            return (positive, negative)

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
        return (out[0], out[1])


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": TimestepKeyframeNode,
    "LatentKeyframe": LatentKeyframeNode,
    "LatentKeyframeGroup": LatentKeyframeGroupNode,
    "LatentKeyframeBatchedGroup": LatentKeyframeBatchedGroupNode,
    "LatentKeyframeTiming": LatentKeyframeInterpolationNode,
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
    # LOOSEControl
    #"ACN_ControlNetLoaderWithLoraAdvanced": ControlNetLoaderWithLoraAdvanced,
    # Deprecated
    "LoadImagesFromDirectory": LoadImagesFromDirectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": "Timestep Keyframe 🛂🅐🅒🅝",
    "LatentKeyframe": "Latent Keyframe 🛂🅐🅒🅝",
    "LatentKeyframeGroup": "Latent Keyframe Group 🛂🅐🅒🅝",
    "LatentKeyframeBatchedGroup": "Latent Keyframe Batched Group 🛂🅐🅒🅝",
    "LatentKeyframeTiming": "Latent Keyframe Interpolation 🛂🅐🅒🅝",
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
    "ACN_SparseCtrlMergedLoaderAdvanced": "Load Merged SparseCtrl Model 🛂🅐🅒🅝",
    "ACN_SparseCtrlIndexMethodNode": "SparseCtrl Index Method 🛂🅐🅒🅝",
    "ACN_SparseCtrlSpreadMethodNode": "SparseCtrl Spread Method 🛂🅐🅒🅝",
    # LOOSEControl
    #"ACN_ControlNetLoaderWithLoraAdvanced": "Load Adv. ControlNet Model w/ LoRA 🛂🅐🅒🅝",
    # Deprecated
    "LoadImagesFromDirectory": "Load Images [DEPRECATED] 🛂🅐🅒🅝",
}
