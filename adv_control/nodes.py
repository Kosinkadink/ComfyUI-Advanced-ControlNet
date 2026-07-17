import comfy.sample

from .nodes_main import (ControlNetLoaderAdvanced, DiffControlNetLoaderAdvanced, AnimaLLLiteLoaderAdvanced,
                         AdvancedControlNetApply, AdvancedControlNetApplySingle)
from .nodes_weight import (DefaultWeights, ScaledSoftMaskedUniversalWeights, ScaledSoftUniversalWeights,
                           SoftControlNetWeightsSD15, CustomControlNetWeightsSD15, CustomControlNetWeightsFlux,
                           CustomControlNetWeightsAnima, SoftT2IAdapterWeights, CustomT2IAdapterWeights, ExtrasMiddleMultNode,
                           AnimaLLLiteExtras)
from .nodes_keyframes import (LatentKeyframeGroupNode, LatentKeyframeInterpolationNode, LatentKeyframeBatchedGroupNode, LatentKeyframeNode,
                              TimestepKeyframeNode, TimestepKeyframeInterpolationNode, TimestepKeyframeFromStrengthListNode)
from .nodes_sparsectrl import SparseCtrlMergedLoaderAdvanced, SparseCtrlLoaderAdvanced, SparseIndexMethodNode, SparseSpreadMethodNode, RgbSparseCtrlPreprocessor, SparseWeightExtras
from .nodes_reference import ReferenceControlNetNode, ReferenceControlFinetune, ReferencePreprocessorNode
from .nodes_plusplus import PlusPlusLoaderAdvanced, PlusPlusLoaderSingle, PlusPlusInputNode
from .nodes_ctrlora import CtrLoRALoader
from .nodes_loosecontrol import ControlNetLoaderWithLoraAdvanced
from .nodes_deprecated import (LoadImagesFromDirectory, ScaledSoftUniversalWeightsDeprecated,
                               SoftControlNetWeightsDeprecated, CustomControlNetWeightsDeprecated, 
                               SoftT2IAdapterWeightsDeprecated, CustomT2IAdapterWeightsDeprecated,
                               AdvancedControlNetApplyDEPR, AdvancedControlNetApplySingleDEPR,
                               ControlNetLoaderAdvancedDEPR, DiffControlNetLoaderAdvancedDEPR)
from .logger import logger


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
    "ACN_AdvancedControlNetApply_v2": AdvancedControlNetApply,
    "ACN_AdvancedControlNetApplySingle_v2": AdvancedControlNetApplySingle,
    # Loaders
    "ACN_ControlNetLoaderAdvanced": ControlNetLoaderAdvanced,
    "ACN_DiffControlNetLoaderAdvanced": DiffControlNetLoaderAdvanced,
    "ACN_AnimaLLLiteLoaderAdvanced": AnimaLLLiteLoaderAdvanced,
    # Weights
    "ACN_ScaledSoftControlNetWeights": ScaledSoftUniversalWeights,
    "ScaledSoftMaskedUniversalWeights": ScaledSoftMaskedUniversalWeights,
    "ACN_SoftControlNetWeightsSD15": SoftControlNetWeightsSD15,
    "ACN_CustomControlNetWeightsSD15": CustomControlNetWeightsSD15,
    "ACN_CustomControlNetWeightsFlux": CustomControlNetWeightsFlux,
    "ACN_CustomControlNetWeightsAnima": CustomControlNetWeightsAnima,
    "ACN_SoftT2IAdapterWeights": SoftT2IAdapterWeights,
    "ACN_CustomT2IAdapterWeights": CustomT2IAdapterWeights,
    "ACN_DefaultUniversalWeights": DefaultWeights,
    "ACN_ExtrasMiddleMult": ExtrasMiddleMultNode,
    "ACN_AnimaLLLiteExtras": AnimaLLLiteExtras,
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
    # CtrLoRA
    "ACN_CtrLoRALoader": CtrLoRALoader,
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
    "ACN_AdvancedControlNetApply": AdvancedControlNetApplyDEPR,
    "ACN_AdvancedControlNetApplySingle": AdvancedControlNetApplySingleDEPR,
    "ControlNetLoaderAdvanced": ControlNetLoaderAdvancedDEPR,
    "DiffControlNetLoaderAdvanced": DiffControlNetLoaderAdvancedDEPR,
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
    "ACN_AdvancedControlNetApply_v2": "Apply Advanced ControlNet 🛂🅐🅒🅝",
    "ACN_AdvancedControlNetApplySingle_v2": "Apply Advanced ControlNet(1) 🛂🅐🅒🅝",
    # Loaders
    "ACN_ControlNetLoaderAdvanced": "Load Advanced ControlNet Model 🛂🅐🅒🅝",
    "ACN_DiffControlNetLoaderAdvanced": "Load Advanced ControlNet Model (diff) 🛂🅐🅒🅝",
    "ACN_AnimaLLLiteLoaderAdvanced": "Load Anima LLLite Model 🛂🅐🅒🅝",
    # Weights
    "ACN_ScaledSoftControlNetWeights": "Scaled Soft Weights 🛂🅐🅒🅝",
    "ScaledSoftMaskedUniversalWeights": "Scaled Soft Masked Weights 🛂🅐🅒🅝",
    "ACN_SoftControlNetWeightsSD15": "ControlNet Soft Weights [SD1.5] 🛂🅐🅒🅝",
    "ACN_CustomControlNetWeightsSD15": "ControlNet Custom Weights [SD1.5] 🛂🅐🅒🅝",
    "ACN_CustomControlNetWeightsFlux": "ControlNet Custom Weights [Flux] 🛂🅐🅒🅝",
    "ACN_CustomControlNetWeightsAnima": "ControlNet Custom Weights [Anima] 🛂🅐🅒🅝",
    "ACN_SoftT2IAdapterWeights": "T2IAdapter Soft Weights 🛂🅐🅒🅝",
    "ACN_CustomT2IAdapterWeights": "T2IAdapter Custom Weights 🛂🅐🅒🅝",
    "ACN_DefaultUniversalWeights": "Default Weights 🛂🅐🅒🅝",
    "ACN_ExtrasMiddleMult": "Middle Weight Extras 🛂🅐🅒🅝",
    "ACN_AnimaLLLiteExtras": "Anima LLLite Extras 🛂🅐🅒🅝",
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
    # CtrLoRA
    "ACN_CtrLoRALoader": "Load CtrLoRA Model 🛂🅐🅒🅝",
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
    "ACN_AdvancedControlNetApply": "Apply Advanced ControlNet 🛂🅐🅒🅝",
    "ACN_AdvancedControlNetApplySingle": "Apply Advanced ControlNet(1) 🛂🅐🅒🅝",
    "ControlNetLoaderAdvanced": "Load Advanced ControlNet Model 🛂🅐🅒🅝",
    "DiffControlNetLoaderAdvanced": "Load Advanced ControlNet Model (diff) 🛂🅐🅒🅝",
}
