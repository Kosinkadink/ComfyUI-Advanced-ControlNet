from comfy_api.latest import io
from torch import Tensor

import folder_paths
import comfy.utils

from .control import load_controlnet, convert_to_advanced, is_advanced_controlnet, is_sd3_advanced_controlnet
from .control_lllite import load_anima_lllite
from .utils import ControlWeights, LatentKeyframeGroup, TimestepKeyframeGroup, AbstractPreprocWrapper

class ControlNetLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_ControlNetLoaderAdvanced',
            display_name='Load Advanced ControlNet Model 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝',
            inputs=[
                io.Combo.Input('cnet', options=folder_paths.get_filename_list("controlnet")),
                io.Custom('TIMESTEP_KEYFRAME').Input('_tk_opt', optional=True)
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ]
        )

    @classmethod
    def execute(cls, cnet,
                        _tk_opt: TimestepKeyframeGroup=None,
                        ):
        controlnet_path = folder_paths.get_full_path("controlnet", cnet)
        controlnet = load_controlnet(controlnet_path, _tk_opt)
        return io.NodeOutput(controlnet,)
    

class DiffControlNetLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_DiffControlNetLoaderAdvanced',
            display_name='Load Advanced ControlNet Model (diff) 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝',
            inputs=[
                io.Model.Input('model'),
                io.Combo.Input('cnet', options=folder_paths.get_filename_list("controlnet")),
                io.Custom('TIMESTEP_KEYFRAME').Input('_tk_opt', optional=True)
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, cnet, model,
                        _tk_opt: TimestepKeyframeGroup=None,
                        ):
        controlnet_path = folder_paths.get_full_path("controlnet", cnet)
        controlnet = load_controlnet(controlnet_path, _tk_opt, model)
        if is_advanced_controlnet(controlnet):
            controlnet.verify_all_weights()
        return io.NodeOutput(controlnet,)

class AnimaLLLiteLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_AnimaLLLiteLoaderAdvanced',
            display_name='Load Anima LLLite Model 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/loaders',
            inputs=[
                io.Combo.Input('model_patch', options=folder_paths.get_filename_list("model_patches")),
                io.Custom('TIMESTEP_KEYFRAME').Input('timestep_kf', optional=True)
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ]
        )

    @classmethod
    def execute(cls, model_patch, timestep_kf: TimestepKeyframeGroup=None):
        model_patch_path = folder_paths.get_full_path_or_raise("model_patches", model_patch)
        return io.NodeOutput(load_anima_lllite(model_patch_path, timestep_keyframe=timestep_kf),)

class AdvancedControlNetApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_AdvancedControlNetApply_v2',
            display_name='Apply Advanced ControlNet 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝',
            inputs=[
                io.Conditioning.Input('positive'),
                io.Conditioning.Input('negative'),
                io.ControlNet.Input('control_net'),
                io.Image.Input('image'),
                io.Float.Input('strength', default=1.0, max=10.0, min=0.0, step=0.01),
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001),
                io.Mask.Input('mask_optional', optional=True),
                io.Custom('TIMESTEP_KEYFRAME').Input('timestep_kf', optional=True),
                io.Custom('LATENT_KEYFRAME').Input('latent_kf_override', optional=True),
                io.Custom('CONTROL_NET_WEIGHTS').Input('weights_override', optional=True),
                io.Vae.Input('vae_optional', optional=True)
            ],
            outputs=[
                io.Conditioning.Output('positive', is_output_list=False),
                io.Conditioning.Output('negative', is_output_list=False)
            ]
        )

    @classmethod
    def execute(cls, positive, negative, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None, vae_optional=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None, control_apply_to_uncond=False, extra_concat=None):
        if strength == 0:
            return io.NodeOutput(positive, negative)
        if extra_concat is None:
            extra_concat = []

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
                        # make sure control_net is not None to avoid confusing error messages
                        if control_net is None:
                            raise Exception("Passed in control_net is None; something must have went wrong when loading it from a Load ControlNet node.")
                        # copy, convert to advanced if needed, and set cond
                        c_net = convert_to_advanced(control_net.copy()).set_cond_hint(control_hint, strength, (start_percent, end_percent), vae_optional, extra_concat)
                        if is_advanced_controlnet(c_net):
                            # disarm node check
                            c_net.disarm()
                            # check for allow_condhint_latents where vae_optional can't handle it itself
                            if c_net.allow_condhint_latents and not c_net.require_vae:
                                if not isinstance(control_hint, AbstractPreprocWrapper):
                                    raise Exception(f"Type '{type(c_net).__name__}' requires proc_IMAGE input via a corresponding preprocessor, but received a normal Image instead.")
                            else:
                                if isinstance(control_hint, AbstractPreprocWrapper) and not c_net.postpone_condhint_latents_check:
                                    raise Exception(f"Type '{type(c_net).__name__}' requires a normal Image input, but received a proc_IMAGE input instead.")
                            # if vae required, verify vae is passed in
                            if c_net.require_vae:
                                # if controlnet can accept preprocced condhint latents and is the case, ignore vae requirement
                                if c_net.allow_condhint_latents and isinstance(control_hint, AbstractPreprocWrapper):
                                    pass
                                elif not vae_optional:
                                    # make sure SD3 ControlNet will get a special message instead of generic type mention
                                    if is_sd3_advanced_controlnet(c_net):
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
        return io.NodeOutput(out[0], out[1])
    

class AdvancedControlNetInpaintingApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_AdvancedControlNetInpaintingApply',
            display_name='Apply Advanced ControlNet Inpainting 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝',
            inputs=[
                io.Conditioning.Input('positive'),
                io.Conditioning.Input('negative'),
                io.ControlNet.Input('control_net'),
                io.Vae.Input('vae'),
                io.Image.Input('image'),
                io.Mask.Input('inpaint_mask'),
                io.Float.Input('strength', default=1.0, max=10.0, min=0.0, step=0.01),
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001),
                io.Mask.Input('effect_mask_optional', optional=True),
                io.Custom('TIMESTEP_KEYFRAME').Input('timestep_kf', optional=True),
                io.Custom('LATENT_KEYFRAME').Input('latent_kf_override', optional=True),
                io.Custom('CONTROL_NET_WEIGHTS').Input('weights_override', optional=True)
            ],
            outputs=[
                io.Conditioning.Output('positive', is_output_list=False),
                io.Conditioning.Output('negative', is_output_list=False)
            ]
        )

    @classmethod
    def execute(cls, positive, negative, control_net, vae, image, inpaint_mask, strength, start_percent, end_percent,
                effect_mask_optional: Tensor=None, timestep_kf: TimestepKeyframeGroup=None,
                latent_kf_override: LatentKeyframeGroup=None, weights_override: ControlWeights=None):
        if not getattr(control_net, "concat_mask", False):
            raise ValueError("The provided ControlNet does not use an inpaint source mask; use Apply Advanced ControlNet instead.")

        source_mask = 1.0 - inpaint_mask.reshape((-1, 1, inpaint_mask.shape[-2], inpaint_mask.shape[-1]))
        mask_apply = comfy.utils.common_upscale(source_mask, image.shape[2], image.shape[1], "bilinear", "center").round()
        image = image * mask_apply.movedim(1, -1).repeat(1, 1, 1, image.shape[3])

        return AdvancedControlNetApply.execute(
            positive=positive,
            negative=negative,
            control_net=control_net,
            image=image,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
            mask_optional=effect_mask_optional,
            vae_optional=vae,
            timestep_kf=timestep_kf,
            latent_kf_override=latent_kf_override,
            weights_override=weights_override,
            extra_concat=[source_mask]
        )


class AdvancedControlNetApplySingle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_AdvancedControlNetApplySingle_v2',
            display_name='Apply Advanced ControlNet(1) 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝',
            inputs=[
                io.Conditioning.Input('conditioning'),
                io.ControlNet.Input('control_net'),
                io.Image.Input('image'),
                io.Float.Input('strength', default=1.0, max=10.0, min=0.0, step=0.01),
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001),
                io.Mask.Input('mask_optional', optional=True),
                io.Custom('TIMESTEP_KEYFRAME').Input('timestep_kf', optional=True),
                io.Custom('LATENT_KEYFRAME').Input('latent_kf_override', optional=True),
                io.Custom('CONTROL_NET_WEIGHTS').Input('weights_override', optional=True),
                io.Vae.Input('vae_optional', optional=True)
            ],
            outputs=[
                io.Conditioning.Output('CONDITIONING', is_output_list=False),
                io.Model.Output('model_opt', is_output_list=False)
            ]
        )

    @classmethod
    def execute(cls, conditioning, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None, vae_optional=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None):
        values = AdvancedControlNetApply.execute(positive=conditioning, negative=None, control_net=control_net, image=image,
                                                          strength=strength, start_percent=start_percent, end_percent=end_percent,
                                                          mask_optional=mask_optional, vae_optional=vae_optional,
                                                          timestep_kf=timestep_kf, latent_kf_override=latent_kf_override, weights_override=weights_override,
                                                          control_apply_to_uncond=True)
        return io.NodeOutput(values.args[0], None)
