import folder_paths
from nodes import VAEEncode

from .utils import TimestepKeyframeGroup
from .control_sparsectrl import SparseMethod, SparseIndexMethod, SparseSettings, SparseSpreadMethod
from .control import load_sparsectrl


# node for SparseCtrl loading
class SparseCtrlLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl"

    def load_controlnet(self, control_net_name: str, use_motion: bool, motion_strength: float, motion_scale: float, sparse_method: SparseMethod=SparseSpreadMethod(), tk_optional: TimestepKeyframeGroup=None):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        sparse_settings = SparseSettings(sparse_method=sparse_method, use_motion=use_motion, motion_strength=motion_strength, motion_scale=motion_scale)
        controlnet = load_sparsectrl(controlnet_path, timestep_keyframe=tk_optional, sparse_settings=sparse_settings)
        return (controlnet,)


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
        idxs = []
        unique_idxs = set()
        # get indeces from string
        str_idxs = [x.strip() for x in indexes.strip().split(",")]
        for str_idx in str_idxs:
            try:
                idx = int(str_idx)
                if idx in unique_idxs:
                    raise ValueError(f"'{idx}' is duplicated; indexes must be unique.")
                idxs.append(idx)
                unique_idxs.add(idx)
            except ValueError:
                raise ValueError(f"'{str_idx}' is not a valid integer index.")
        if len(idxs) == 0:
            raise ValueError(f"No indexes were listed in Sparse Index Method.")
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


class VAEEncodePreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "vae": ("VAE", )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("latent_IMAGE",)
    FUNCTION = "preprocess_images"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/preprocess"

    def preprocess_images(self, vae, image):
        image = VAEEncode.vae_encode_crop_pixels(image)
        encoded = vae.encode(image[:,:,:,:3])
        encoded = encoded.movedim(1,-1)
        return (encoded,)
