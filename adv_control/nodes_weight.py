from comfy_api.latest import io
from torch import Tensor
import torch
from .utils import TimestepKeyframe, TimestepKeyframeGroup, ControlWeights, Extras, get_properly_arranged_t2i_weights, linear_conversion
from .control_lllite import AnimaLLLiteConst

WEIGHTS_RETURN_NAMES = ("CN_WEIGHTS", "TK_SHORTCUT")

class DefaultWeights(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_DefaultUniversalWeights',
            display_name='Default Weights 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights',
            inputs=[
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, cn_extras: dict[str]={}):
        weights = ControlWeights.default(extras=cn_extras)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class ScaledSoftMaskedUniversalWeights(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ScaledSoftMaskedUniversalWeights',
            display_name='Scaled Soft Masked Weights 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights',
            inputs=[
                io.Mask.Input('mask'),
                io.Float.Input('min_base_multiplier', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('max_base_multiplier', default=1.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, mask: Tensor, min_base_multiplier: float, max_base_multiplier: float, lock_min=False, lock_max=False,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        # normalize mask
        mask = mask.clone()
        x_min = 0.0 if lock_min else mask.min()
        x_max = 1.0 if lock_max else mask.max()
        if x_min == x_max:
            mask = torch.ones_like(mask) * max_base_multiplier
        else:
            mask = linear_conversion(mask, x_min, x_max, min_base_multiplier, max_base_multiplier)
        weights = ControlWeights.universal_mask(weight_mask=mask, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class ScaledSoftUniversalWeights(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_ScaledSoftControlNetWeights',
            display_name='Scaled Soft Weights 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights',
            inputs=[
                io.Float.Input('base_multiplier', default=0.825, max=1.0, min=0.0, step=0.001),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, base_multiplier, uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = ControlWeights.universal(base_multiplier=base_multiplier, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class SoftControlNetWeightsSD15(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_SoftControlNetWeightsSD15',
            display_name='ControlNet Soft Weights [SD1.5] 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet',
            inputs=[
                io.Float.Input('output_0', default=0.09941396206337118, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_1', default=0.12050177219802567, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_2', default=0.14606275417942507, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_3', default=0.17704576264172736, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_4', default=0.214600924414215, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_5', default=0.26012233262329093, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_6', default=0.3152997971191405, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_7', default=0.3821815722656249, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_8', default=0.4632503906249999, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_9', default=0.561515625, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_10', default=0.6806249999999999, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_11', default=0.825, max=10.0, min=0.0, step=0.001),
                io.Float.Input('middle_0', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, output_0, output_1, output_2, output_3, output_4, output_5, output_6,
                     output_7, output_8, output_9, output_10, output_11, middle_0,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        return CustomControlNetWeightsSD15.execute(
                                                        output_0=output_0, output_1=output_1, output_2=output_2, output_3=output_3,
                                                        output_4=output_4, output_5=output_5, output_6=output_6, output_7=output_7,
                                                        output_8=output_8, output_9=output_9, output_10=output_10, output_11=output_11,
                                                        middle_0=middle_0,
                                                        uncond_multiplier=uncond_multiplier, cn_extras=cn_extras)

class CustomControlNetWeightsSD15(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_CustomControlNetWeightsSD15',
            display_name='ControlNet Custom Weights [SD1.5] 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet',
            inputs=[
                io.Float.Input('output_0', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_1', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_2', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_3', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_4', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_5', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_6', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_7', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_8', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_9', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_10', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('output_11', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('middle_0', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, output_0, output_1, output_2, output_3, output_4, output_5, output_6,
                     output_7, output_8, output_9, output_10, output_11, middle_0,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights_output = [output_0, output_1, output_2, output_3, output_4, output_5, output_6,
                          output_7, output_8, output_9, output_10, output_11]
        weights_middle = [middle_0]
        weights = ControlWeights.controlnet(weights_output=weights_output, weights_middle=weights_middle, uncond_multiplier=uncond_multiplier,
                                            extras=cn_extras, disable_applied_to=True)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class CustomControlNetWeightsFlux(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_CustomControlNetWeightsFlux',
            display_name='ControlNet Custom Weights [Flux] 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet',
            inputs=[
                io.Float.Input('input_0', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_1', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_2', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_3', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_4', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_5', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_6', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_7', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_8', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_9', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_10', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_11', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_12', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_13', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_14', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_15', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_16', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_17', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_18', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
                     input_7, input_8, input_9, input_10, input_11, input_12, input_13,
                     input_14, input_15, input_16, input_17, input_18,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights_input = [input_0, input_1, input_2, input_3, input_4, input_5,
                         input_6, input_7, input_8, input_9, input_10, input_11,
                         input_12, input_13, input_14, input_15, input_16, input_17, input_18]
        weights = ControlWeights.controlnet(weights_input=weights_input, uncond_multiplier=uncond_multiplier, extras=cn_extras, disable_applied_to=True)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class CustomControlNetWeightsAnima(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_CustomControlNetWeightsAnima',
            display_name='ControlNet Custom Weights [Anima] 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet',
            inputs=[
                io.Float.Input('block_0', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_1', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_2', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_3', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_4', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_5', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_6', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_7', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_8', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_9', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_10', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_11', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_12', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_13', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_14', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_15', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_16', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_17', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_18', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_19', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_20', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_21', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_22', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_23', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_24', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_25', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_26', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('block_27', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )

    @classmethod
    def execute(cls, uncond_multiplier: float=1.0, cn_extras: dict[str]={}, **kwargs):
        weights = [kwargs[f"block_{index}"] for index in range(28)]
        control_weights = ControlWeights.controllllite(
            weights_input=weights,
            uncond_multiplier=uncond_multiplier,
            extras=cn_extras,
        )
        return io.NodeOutput(control_weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=control_weights)))

class SoftT2IAdapterWeights(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_SoftT2IAdapterWeights',
            display_name='T2IAdapter Soft Weights 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights/T2IAdapter',
            inputs=[
                io.Float.Input('input_0', default=0.25, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_1', default=0.62, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_2', default=0.825, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_3', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, input_0, input_1, input_2, input_3,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        return CustomT2IAdapterWeights.execute(input_0=input_0, input_1=input_1, input_2=input_2, input_3=input_3,
                                                    uncond_multiplier=uncond_multiplier, cn_extras=cn_extras)

class CustomT2IAdapterWeights(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_CustomT2IAdapterWeights',
            display_name='T2IAdapter Custom Weights 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights/T2IAdapter',
            inputs=[
                io.Float.Input('input_0', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_1', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_2', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('input_3', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, input_0, input_1, input_2, input_3,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = [input_0, input_1, input_2, input_3]
        weights = get_properly_arranged_t2i_weights(weights)
        weights = ControlWeights.t2iadapter(weights_input=weights, uncond_multiplier=uncond_multiplier, extras=cn_extras, disable_applied_to=True)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class ExtrasMiddleMultNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_ExtrasMiddleMult',
            display_name='Middle Weight Extras 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights/extras',
            inputs=[
                io.Float.Input('middle_mult', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CN_WEIGHTS_EXTRAS').Output('cn_extras', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, middle_mult: float, cn_extras: dict[str]={}):
        cn_extras = cn_extras.copy()
        cn_extras[Extras.MIDDLE_MULT] = middle_mult
        return io.NodeOutput(cn_extras,)

class AnimaLLLiteExtras(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_AnimaLLLiteExtras',
            display_name='Anima LLLite Extras 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/weights/extras',
            inputs=[
                io.Mask.Input('inpaint_mask'),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CN_WEIGHTS_EXTRAS').Output('cn_extras', is_output_list=False)
            ]
        )

    @classmethod
    def execute(cls, inpaint_mask: Tensor, cn_extras: dict[str]={}):
        cn_extras = cn_extras.copy()
        cn_extras[AnimaLLLiteConst.INPAINT_MASK] = inpaint_mask.clone()
        return io.NodeOutput(cn_extras,)
