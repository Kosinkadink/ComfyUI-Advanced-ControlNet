from torch import Tensor

import folder_paths
from nodes import VAEEncode
import comfy.utils
from comfy.sd import VAE

from .utils import TimestepKeyframeGroup
from .control_sparsectrl import SparseMethod, SparseIndexMethod, SparseSettings, SparseSpreadMethod, PreprocSparseRGBWrapper, SparseConst, SparseContextAware, get_idx_list_from_str
from .control import load_sparsectrl, load_controlnet, ControlNetAdvanced, SparseCtrlAdvanced


# node for SparseCtrl loading
class SparseCtrlLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sparsectrl_name": (folder_paths.get_filename_list("controlnet"), ),
                "use_motion": ("BOOLEAN", {"default": True}, ),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            },
            "optional": {
                "sparse_method": ("SPARSE_METHOD", ),
                "tk_optional": ("TIMESTEP_KEYFRAME", ),
                "context_aware": (SparseContextAware.LIST, ),
                "sparse_hint_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "sparse_nonhint_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "sparse_mask_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl"

    def load_controlnet(self, sparsectrl_name: str, use_motion: bool, motion_strength: float, motion_scale: float, sparse_method: SparseMethod=SparseSpreadMethod(), tk_optional: TimestepKeyframeGroup=None,
                        context_aware=SparseContextAware.NEAREST_HINT, sparse_hint_mult=1.0, sparse_nonhint_mult=1.0, sparse_mask_mult=1.0):
        sparsectrl_path = folder_paths.get_full_path("controlnet", sparsectrl_name)
        sparse_settings = SparseSettings(sparse_method=sparse_method, use_motion=use_motion, motion_strength=motion_strength, motion_scale=motion_scale,
                                         context_aware=context_aware,
                                         sparse_mask_mult=sparse_mask_mult, sparse_hint_mult=sparse_hint_mult, sparse_nonhint_mult=sparse_nonhint_mult)
        sparsectrl = load_sparsectrl(sparsectrl_path, timestep_keyframe=tk_optional, sparse_settings=sparse_settings)
        return (sparsectrl,)


class SparseCtrlMergedLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sparsectrl_name": (folder_paths.get_filename_list("controlnet"), ),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
                "use_motion": ("BOOLEAN", {"default": True}, ),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            },
            "optional": {
                "sparse_method": ("SPARSE_METHOD", ),
                "tk_optional": ("TIMESTEP_KEYFRAME", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/experimental"

    def load_controlnet(self, sparsectrl_name: str, control_net_name: str, use_motion: bool, motion_strength: float, motion_scale: float, sparse_method: SparseMethod=SparseSpreadMethod(), tk_optional: TimestepKeyframeGroup=None):
        sparsectrl_path = folder_paths.get_full_path("controlnet", sparsectrl_name)
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        sparse_settings = SparseSettings(sparse_method=sparse_method, use_motion=use_motion, motion_strength=motion_strength, motion_scale=motion_scale, merged=True)
        # first, load normal controlnet
        controlnet = load_controlnet(controlnet_path, timestep_keyframe=tk_optional)
        # confirm that controlnet is ControlNetAdvanced
        if controlnet is None or type(controlnet) != ControlNetAdvanced:
            raise ValueError(f"controlnet_path must point to a normal ControlNet, but instead: {type(controlnet).__name__}")
        # next, load sparsectrl, making sure to load motion portion
        sparsectrl = load_sparsectrl(sparsectrl_path, timestep_keyframe=tk_optional, sparse_settings=SparseSettings.default())
        # now, combine state dicts
        new_state_dict = controlnet.control_model.state_dict()
        for key, value in sparsectrl.control_model.motion_holder.motion_wrapper.state_dict().items():
            new_state_dict[key] = value
        # now, reload sparsectrl with real settings
        sparsectrl = load_sparsectrl(sparsectrl_path, controlnet_data=new_state_dict, timestep_keyframe=tk_optional, sparse_settings=sparse_settings)
        return (sparsectrl,)


class SparseIndexMethodNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "indexes": ("STRING", {"default": "0"}),
            }
        }
    
    RETURN_TYPES = ("SPARSE_METHOD",)
    FUNCTION = "get_method"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl"

    def get_method(self, indexes: str):
        idxs = get_idx_list_from_str(indexes)
        return (SparseIndexMethod(idxs),)


class SparseSpreadMethodNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "spread": (SparseSpreadMethod.LIST,),
            }
        }
    
    RETURN_TYPES = ("SPARSE_METHOD",)
    FUNCTION = "get_method"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl"

    def get_method(self, spread: str):
        return (SparseSpreadMethod(spread=spread),)


class RgbSparseCtrlPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "vae": ("VAE", ),
                "latent_size": ("LATENT", ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("proc_IMAGE",)
    FUNCTION = "preprocess_images"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/preprocess"

    def preprocess_images(self, vae: VAE, image: Tensor, latent_size: Tensor):
        # first, resize image to match latents
        image = image.movedim(-1,1)
        image = comfy.utils.common_upscale(image, latent_size["samples"].shape[3] * 8, latent_size["samples"].shape[2] * 8, 'nearest-exact', "center")
        image = image.movedim(1,-1)
        # then, vae encode
        try:
            image = vae.vae_encode_crop_pixels(image)
        except Exception:
            image = VAEEncode.vae_encode_crop_pixels(image)
        encoded = vae.encode(image[:,:,:,:3])
        return (PreprocSparseRGBWrapper(condhint=encoded),)


class SparseWeightExtras:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
                "sparse_hint_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "sparse_nonhint_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "sparse_mask_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CN_WEIGHTS_EXTRAS", )
    RETURN_NAMES = ("cn_extras", )
    FUNCTION = "create_weight_extras"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/extras"

    def create_weight_extras(self, cn_extras: dict[str]={}, sparse_hint_mult=1.0, sparse_nonhint_mult=1.0, sparse_mask_mult=1.0):
        cn_extras = cn_extras.copy()
        cn_extras[SparseConst.HINT_MULT] = sparse_hint_mult
        cn_extras[SparseConst.NONHINT_MULT] = sparse_nonhint_mult
        cn_extras[SparseConst.MASK_MULT] = sparse_mask_mult
        return (cn_extras, )
