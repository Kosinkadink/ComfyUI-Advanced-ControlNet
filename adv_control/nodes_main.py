from torch import Tensor

import folder_paths
from comfy.model_patcher import ModelPatcher

from .control import load_controlnet, convert_to_advanced, is_advanced_controlnet, is_sd3_advanced_controlnet
from .utils import ControlWeights, LatentKeyframeGroup, TimestepKeyframeGroup, AbstractPreprocWrapper, BIGMAX

from .logger import logger


class ControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cnet": (folder_paths.get_filename_list("controlnet"), ),
            },
            "optional": {
                "_tk_opt": ("TIMESTEP_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def load_controlnet(self, cnet,
                        _tk_opt: TimestepKeyframeGroup=None,
                        ):
        controlnet_path = folder_paths.get_full_path("controlnet", cnet)
        controlnet = load_controlnet(controlnet_path, _tk_opt)
        return (controlnet,)
    

class DiffControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cnet": (folder_paths.get_filename_list("controlnet"), )
            },
            "optional": {
                "_tk_opt": ("TIMESTEP_KEYFRAME", ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def load_controlnet(self, cnet, model,
                        _tk_opt: TimestepKeyframeGroup=None,
                        ):
        controlnet_path = folder_paths.get_full_path("controlnet", cnet)
        controlnet = load_controlnet(controlnet_path, _tk_opt, model)
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
                "vae_optional": ("VAE",),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None, vae_optional=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None, control_apply_to_uncond=False):
        if strength == 0:
            return (positive, negative)

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
                        c_net = convert_to_advanced(control_net.copy()).set_cond_hint(control_hint, strength, (start_percent, end_percent), vae_optional)
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
        return (out[0], out[1])
    

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
                "vae_optional": ("VAE",),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","MODEL",)
    RETURN_NAMES = ("CONDITIONING", "model_opt")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def apply_controlnet(self, conditioning, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None, vae_optional=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None):
        values = AdvancedControlNetApply.apply_controlnet(self, positive=conditioning, negative=None, control_net=control_net, image=image,
                                                          strength=strength, start_percent=start_percent, end_percent=end_percent,
                                                          mask_optional=mask_optional, vae_optional=vae_optional,
                                                          timestep_kf=timestep_kf, latent_kf_override=latent_kf_override, weights_override=weights_override,
                                                          control_apply_to_uncond=True)
        return (values[0],)
