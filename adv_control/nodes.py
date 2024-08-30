import numpy as np
from torch import Tensor

import folder_paths
import comfy.sample
from comfy.model_patcher import ModelPatcher

from .control import load_controlnet, convert_to_advanced, is_advanced_controlnet, is_sd3_advanced_controlnet
from .utils import ControlWeights, LatentKeyframeGroup, TimestepKeyframeGroup, AbstractPreprocWrapper, BIGMAX
from .nodes_weight import (DefaultWeights, ScaledSoftMaskedUniversalWeights, ScaledSoftUniversalWeights,
                           SoftControlNetWeightsSD15, CustomControlNetWeightsSD15, CustomControlNetWeightsFlux,
                           SoftT2IAdapterWeights, CustomT2IAdapterWeights)
from .nodes_keyframes import (LatentKeyframeGroupNode, LatentKeyframeInterpolationNode, LatentKeyframeBatchedGroupNode, LatentKeyframeNode,
                              TimestepKeyframeNode, TimestepKeyframeInterpolationNode, TimestepKeyframeFromStrengthListNode)
from .nodes_sparsectrl import SparseCtrlMergedLoaderAdvanced, SparseCtrlLoaderAdvanced, SparseIndexMethodNode, SparseSpreadMethodNode, RgbSparseCtrlPreprocessor, SparseWeightExtras
from .nodes_reference import ReferenceControlNetNode, ReferenceControlFinetune, ReferencePreprocessorNode
from .nodes_plusplus import PlusPlusLoaderAdvanced, PlusPlusLoaderSingle, PlusPlusInputNode
from .nodes_loosecontrol import ControlNetLoaderWithLoraAdvanced
from .nodes_deprecated import (LoadImagesFromDirectory, ScaledSoftUniversalWeightsDeprecated,
                               SoftControlNetWeightsDeprecated, CustomControlNetWeightsDeprecated, 
                               SoftT2IAdapterWeightsDeprecated, CustomT2IAdapterWeightsDeprecated)
from .logger import logger

from .sampling import acn_sample_factory
# inject sample functions
comfy.sample.sample = acn_sample_factory(comfy.sample.sample)
comfy.sample.sample_custom = acn_sample_factory(comfy.sample.sample_custom, is_custom=True)


class ControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            },
            "optional": {
                "tk_optional": ("TIMESTEP_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def load_controlnet(self, control_net_name,
                        tk_optional: TimestepKeyframeGroup=None,
                        timestep_keyframe: TimestepKeyframeGroup=None,
                        ):
        if timestep_keyframe is not None: # backwards compatibility
            tk_optional = timestep_keyframe
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, tk_optional)
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
                "tk_optional": ("TIMESTEP_KEYFRAME", ),
                "autosize": ("ACNAUTOSIZE", {"padding": 160}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def load_controlnet(self, control_net_name, model,
                        tk_optional: TimestepKeyframeGroup=None,
                        timestep_keyframe: TimestepKeyframeGroup=None
                        ):
        if timestep_keyframe is not None: # backwards compatibility
            tk_optional = timestep_keyframe
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, tk_optional, model)
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
                "vae_optional": ("VAE",),
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","MODEL",)
    RETURN_NAMES = ("positive", "negative", "model_opt")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None, model_optional: ModelPatcher=None, vae_optional=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None, control_apply_to_uncond=False):
        if strength == 0:
            return (positive, negative, model_optional)
        if model_optional:
            model_optional = model_optional.clone()

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            if conditioning is not None:
                for t in conditioning:
                    d = t[1].copy()

                    prev_cnet = d.get('control', None)
                    if prev_cnet in cnets:
                        c_net = cnets[prev_cnet]
                    else:
                        # copy, convert to advanced if needed, and set cond
                        c_net = convert_to_advanced(control_net.copy()).set_cond_hint(control_hint, strength, (start_percent, end_percent), vae_optional)
                        if is_advanced_controlnet(c_net):
                            # disarm node check
                            c_net.disarm()
                            # if model required, verify model is passed in, and if so patch it
                            if c_net.require_model:
                                if not model_optional:
                                    raise Exception(f"Type '{type(c_net).__name__}' requires model_optional input, but got None.")
                                c_net.patch_model(model=model_optional)
                            # if vae required, verify vae is passed in
                            if c_net.require_vae:
                                # if controlnet can accept preprocced condhint latents and is the case, ignore vae requirement
                                if c_net.allow_condhint_latents and isinstance(control_hint, AbstractPreprocWrapper):
                                    pass
                                elif not vae_optional:
                                    # make sure SD3 ControlNet will get a special message instead of generic type mention
                                    if is_sd3_advanced_controlnet:
                                        raise Exception(f"SD3 ControlNet requires vae_optional input, but got None.")
                                    else:
                                        raise Exception(f"Type '{type(c_net).__name__}' requires vae_optional input, but got None.")
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
                    d['control_apply_to_uncond'] = control_apply_to_uncond
                    n = [t[0], d]
                    c.append(n)
            out.append(c)
        return (out[0], out[1], model_optional)
    

class AdvancedControlNetApplySingle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
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
                "vae_optional": ("VAE",),
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","MODEL",)
    RETURN_NAMES = ("CONDITIONING", "model_opt")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def apply_controlnet(self, conditioning, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None, model_optional: ModelPatcher=None, vae_optional=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None):
        values = AdvancedControlNetApply.apply_controlnet(self, positive=conditioning, negative=None, control_net=control_net, image=image,
                                                          strength=strength, start_percent=start_percent, end_percent=end_percent,
                                                          mask_optional=mask_optional, model_optional=model_optional, vae_optional=vae_optional,
                                                          timestep_kf=timestep_kf, latent_kf_override=latent_kf_override, weights_override=weights_override,
                                                          control_apply_to_uncond=True)
        return (values[0], values[2])


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
    "ACN_AdvancedControlNetApplySingle": AdvancedControlNetApplySingle,
    # Loaders
    "ControlNetLoaderAdvanced": ControlNetLoaderAdvanced,
    "DiffControlNetLoaderAdvanced": DiffControlNetLoaderAdvanced,
    # Weights
    "ACN_ScaledSoftControlNetWeights": ScaledSoftUniversalWeights,
    "ScaledSoftMaskedUniversalWeights": ScaledSoftMaskedUniversalWeights,
    "ACN_SoftControlNetWeightsSD15": SoftControlNetWeightsSD15,
    "ACN_CustomControlNetWeightsSD15": CustomControlNetWeightsSD15,
    "ACN_CustomControlNetWeightsFlux": CustomControlNetWeightsFlux,
    "ACN_SoftT2IAdapterWeights": SoftT2IAdapterWeights,
    "ACN_CustomT2IAdapterWeights": CustomT2IAdapterWeights,
    "ACN_DefaultUniversalWeights": DefaultWeights,
    # SparseCtrl
    "ACN_SparseCtrlRGBPreprocessor": RgbSparseCtrlPreprocessor,
    "ACN_SparseCtrlLoaderAdvanced": SparseCtrlLoaderAdvanced,
    "ACN_SparseCtrlMergedLoaderAdvanced": SparseCtrlMergedLoaderAdvanced,
    "ACN_SparseCtrlIndexMethodNode": SparseIndexMethodNode,
    "ACN_SparseCtrlSpreadMethodNode": SparseSpreadMethodNode,
    "ACN_SparseCtrlWeightExtras": SparseWeightExtras,
    # ControlNet++
    "ACN_ControlNet++LoaderSingle": PlusPlusLoaderSingle,
    "ACN_ControlNet++LoaderAdvanced": PlusPlusLoaderAdvanced,
    "ACN_ControlNet++InputNode": PlusPlusInputNode,
    # Reference
    "ACN_ReferencePreprocessor": ReferencePreprocessorNode,
    "ACN_ReferenceControlNet": ReferenceControlNetNode,
    "ACN_ReferenceControlNetFinetune": ReferenceControlFinetune,
    # LOOSEControl
    #"ACN_ControlNetLoaderWithLoraAdvanced": ControlNetLoaderWithLoraAdvanced,
    # Deprecated
    "LoadImagesFromDirectory": LoadImagesFromDirectory,
    "ScaledSoftControlNetWeights": ScaledSoftUniversalWeightsDeprecated,
    "SoftControlNetWeights": SoftControlNetWeightsDeprecated,
    "CustomControlNetWeights": CustomControlNetWeightsDeprecated,
    "SoftT2IAdapterWeights": SoftT2IAdapterWeightsDeprecated,
    "CustomT2IAdapterWeights": CustomT2IAdapterWeightsDeprecated,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": "Timestep Keyframe 🛂🅐🅒🅝",
    "ACN_TimestepKeyframeInterpolation": "Timestep Keyframe Interp. 🛂🅐🅒🅝",
    "ACN_TimestepKeyframeFromStrengthList": "Timestep Keyframe From List 🛂🅐🅒🅝",
    "LatentKeyframe": "Latent Keyframe 🛂🅐🅒🅝",
    "LatentKeyframeTiming": "Latent Keyframe Interp. 🛂🅐🅒🅝",
    "LatentKeyframeBatchedGroup": "Latent Keyframe From List 🛂🅐🅒🅝",
    "LatentKeyframeGroup": "Latent Keyframe Group 🛂🅐🅒🅝",
    # Conditioning
    "ACN_AdvancedControlNetApply": "Apply Advanced ControlNet 🛂🅐🅒🅝",
    "ACN_AdvancedControlNetApplySingle": "Apply Advanced ControlNet(1) 🛂🅐🅒🅝",
    # Loaders
    "ControlNetLoaderAdvanced": "Load Advanced ControlNet Model 🛂🅐🅒🅝",
    "DiffControlNetLoaderAdvanced": "Load Advanced ControlNet Model (diff) 🛂🅐🅒🅝",
    # Weights
    "ACN_ScaledSoftControlNetWeights": "Scaled Soft Weights 🛂🅐🅒🅝",
    "ScaledSoftMaskedUniversalWeights": "Scaled Soft Masked Weights 🛂🅐🅒🅝",
    "ACN_SoftControlNetWeightsSD15": "ControlNet Soft Weights [SD1.5] 🛂🅐🅒🅝",
    "ACN_CustomControlNetWeightsSD15": "ControlNet Custom Weights [SD1.5] 🛂🅐🅒🅝",
    "ACN_CustomControlNetWeightsFlux": "ControlNet Custom Weights [Flux] 🛂🅐🅒🅝",
    "ACN_SoftT2IAdapterWeights": "T2IAdapter Soft Weights 🛂🅐🅒🅝",
    "ACN_CustomT2IAdapterWeights": "T2IAdapter Custom Weights 🛂🅐🅒🅝",
    "ACN_DefaultUniversalWeights": "Default Weights 🛂🅐🅒🅝",
    # SparseCtrl
    "ACN_SparseCtrlRGBPreprocessor": "RGB SparseCtrl 🛂🅐🅒🅝",
    "ACN_SparseCtrlLoaderAdvanced": "Load SparseCtrl Model 🛂🅐🅒🅝",
    "ACN_SparseCtrlMergedLoaderAdvanced": "🧪Load Merged SparseCtrl Model 🛂🅐🅒🅝",
    "ACN_SparseCtrlIndexMethodNode": "SparseCtrl Index Method 🛂🅐🅒🅝",
    "ACN_SparseCtrlSpreadMethodNode": "SparseCtrl Spread Method 🛂🅐🅒🅝",
    "ACN_SparseCtrlWeightExtras": "SparseCtrl Weight Extras 🛂🅐🅒🅝",
    # ControlNet++
    "ACN_ControlNet++LoaderSingle": "Load ControlNet++ Model (Single) 🛂🅐🅒🅝",
    "ACN_ControlNet++LoaderAdvanced": "Load ControlNet++ Model (Multi) 🛂🅐🅒🅝",
    "ACN_ControlNet++InputNode": "ControlNet++ Input 🛂🅐🅒🅝",
    # Reference
    "ACN_ReferencePreprocessor": "Reference Preproccessor 🛂🅐🅒🅝",
    "ACN_ReferenceControlNet": "Reference ControlNet 🛂🅐🅒🅝",
    "ACN_ReferenceControlNetFinetune": "Reference ControlNet (Finetune) 🛂🅐🅒🅝",
    # LOOSEControl
    #"ACN_ControlNetLoaderWithLoraAdvanced": "Load Adv. ControlNet Model w/ LoRA 🛂🅐🅒🅝",
    # Deprecated
    "LoadImagesFromDirectory": "🚫Load Images [DEPRECATED] 🛂🅐🅒🅝",
    "ScaledSoftControlNetWeights": "Scaled Soft Weights 🛂🅐🅒🅝",
    "SoftControlNetWeights": "ControlNet Soft Weights 🛂🅐🅒🅝",
    "CustomControlNetWeights": "ControlNet Custom Weights 🛂🅐🅒🅝",
    "SoftT2IAdapterWeights": "T2IAdapter Soft Weights 🛂🅐🅒🅝",
    "CustomT2IAdapterWeights": "T2IAdapter Custom Weights 🛂🅐🅒🅝",
}
