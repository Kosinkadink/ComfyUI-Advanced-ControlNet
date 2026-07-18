from comfy_api.latest import ComfyExtension, io

from .nodes_main import (ControlNetLoaderAdvanced, DiffControlNetLoaderAdvanced, AnimaLLLiteLoaderAdvanced,
                         AdvancedControlNetApply, AdvancedControlNetInpaintingApply, AdvancedControlNetApplySingle)
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
from .nodes_deprecated import (LoadImagesFromDirectory, ScaledSoftUniversalWeightsDeprecated,
                               SoftControlNetWeightsDeprecated, CustomControlNetWeightsDeprecated, 
                               SoftT2IAdapterWeightsDeprecated, CustomT2IAdapterWeightsDeprecated,
                               AdvancedControlNetApplyDEPR, AdvancedControlNetApplySingleDEPR,
                               ControlNetLoaderAdvancedDEPR, DiffControlNetLoaderAdvancedDEPR)




class AdvancedControlNetExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TimestepKeyframeNode,
            TimestepKeyframeInterpolationNode,
            TimestepKeyframeFromStrengthListNode,
            LatentKeyframeNode,
            LatentKeyframeInterpolationNode,
            LatentKeyframeBatchedGroupNode,
            LatentKeyframeGroupNode,
            AdvancedControlNetApply,
            AdvancedControlNetInpaintingApply,
            AdvancedControlNetApplySingle,
            ControlNetLoaderAdvanced,
            DiffControlNetLoaderAdvanced,
            AnimaLLLiteLoaderAdvanced,
            ScaledSoftUniversalWeights,
            ScaledSoftMaskedUniversalWeights,
            SoftControlNetWeightsSD15,
            CustomControlNetWeightsSD15,
            CustomControlNetWeightsFlux,
            CustomControlNetWeightsAnima,
            SoftT2IAdapterWeights,
            CustomT2IAdapterWeights,
            DefaultWeights,
            ExtrasMiddleMultNode,
            AnimaLLLiteExtras,
            RgbSparseCtrlPreprocessor,
            SparseCtrlLoaderAdvanced,
            SparseCtrlMergedLoaderAdvanced,
            SparseIndexMethodNode,
            SparseSpreadMethodNode,
            SparseWeightExtras,
            PlusPlusLoaderSingle,
            PlusPlusLoaderAdvanced,
            PlusPlusInputNode,
            CtrLoRALoader,
            ReferencePreprocessorNode,
            ReferenceControlNetNode,
            ReferenceControlFinetune,
            LoadImagesFromDirectory,
            ScaledSoftUniversalWeightsDeprecated,
            SoftControlNetWeightsDeprecated,
            CustomControlNetWeightsDeprecated,
            SoftT2IAdapterWeightsDeprecated,
            CustomT2IAdapterWeightsDeprecated,
            AdvancedControlNetApplyDEPR,
            AdvancedControlNetApplySingleDEPR,
            ControlNetLoaderAdvancedDEPR,
            DiffControlNetLoaderAdvancedDEPR
        ]
