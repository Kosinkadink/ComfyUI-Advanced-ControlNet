from comfy_api.latest import io
from torch import Tensor

import folder_paths
from nodes import VAEEncode
import comfy.utils
from comfy.sd import VAE

from .utils import TimestepKeyframeGroup
from .control_sparsectrl import SparseMethod, SparseIndexMethod, SparseSettings, SparseSpreadMethod, PreprocSparseRGBWrapper, SparseConst, SparseContextAware, get_idx_list_from_str
from .control import load_sparsectrl, load_controlnet, ControlNetAdvanced

# node for SparseCtrl loading
class SparseCtrlLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_SparseCtrlLoaderAdvanced',
            display_name='Load SparseCtrl Model 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl',
            inputs=[
                io.Combo.Input('sparsectrl_name', options=folder_paths.get_filename_list("controlnet")),
                io.Boolean.Input('use_motion', default=True),
                io.Float.Input('motion_strength', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('motion_scale', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Custom('SPARSE_METHOD').Input('sparse_method', optional=True),
                io.Custom('TIMESTEP_KEYFRAME').Input('tk_optional', optional=True),
                io.Combo.Input('context_aware', optional=True, options=['nearest_hint', 'off']),
                io.Float.Input('sparse_hint_mult', optional=True, default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('sparse_nonhint_mult', optional=True, default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('sparse_mask_mult', optional=True, default=1.0, max=10.0, min=0.0, step=0.001)
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, sparsectrl_name: str, use_motion: bool, motion_strength: float, motion_scale: float, sparse_method: SparseMethod=SparseSpreadMethod(), tk_optional: TimestepKeyframeGroup=None,
                        context_aware=SparseContextAware.NEAREST_HINT, sparse_hint_mult=1.0, sparse_nonhint_mult=1.0, sparse_mask_mult=1.0):
        sparsectrl_path = folder_paths.get_full_path("controlnet", sparsectrl_name)
        sparse_settings = SparseSettings(sparse_method=sparse_method, use_motion=use_motion, motion_strength=motion_strength, motion_scale=motion_scale,
                                         context_aware=context_aware,
                                         sparse_mask_mult=sparse_mask_mult, sparse_hint_mult=sparse_hint_mult, sparse_nonhint_mult=sparse_nonhint_mult)
        sparsectrl = load_sparsectrl(sparsectrl_path, timestep_keyframe=tk_optional, sparse_settings=sparse_settings)
        return io.NodeOutput(sparsectrl,)

class SparseCtrlMergedLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_SparseCtrlMergedLoaderAdvanced',
            display_name='🧪Load Merged SparseCtrl Model 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/experimental',
            inputs=[
                io.Combo.Input('sparsectrl_name', options=folder_paths.get_filename_list("controlnet")),
                io.Combo.Input('control_net_name', options=folder_paths.get_filename_list("controlnet")),
                io.Boolean.Input('use_motion', default=True),
                io.Float.Input('motion_strength', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('motion_scale', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Custom('SPARSE_METHOD').Input('sparse_method', optional=True),
                io.Custom('TIMESTEP_KEYFRAME').Input('tk_optional', optional=True)
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, sparsectrl_name: str, control_net_name: str, use_motion: bool, motion_strength: float, motion_scale: float, sparse_method: SparseMethod=SparseSpreadMethod(), tk_optional: TimestepKeyframeGroup=None):
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
        return io.NodeOutput(sparsectrl,)

class SparseIndexMethodNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_SparseCtrlIndexMethodNode',
            display_name='SparseCtrl Index Method 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl',
            inputs=[
                io.String.Input('indexes', default='0')
            ],
            outputs=[
                io.Custom('SPARSE_METHOD').Output('SPARSE_METHOD', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, indexes: str):
        idxs = get_idx_list_from_str(indexes)
        return io.NodeOutput(SparseIndexMethod(idxs),)

class SparseSpreadMethodNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_SparseCtrlSpreadMethodNode',
            display_name='SparseCtrl Spread Method 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl',
            inputs=[
                io.Combo.Input('spread', options=['uniform', 'starting', 'ending', 'center'])
            ],
            outputs=[
                io.Custom('SPARSE_METHOD').Output('SPARSE_METHOD', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, spread: str):
        return io.NodeOutput(SparseSpreadMethod(spread=spread),)

class RgbSparseCtrlPreprocessor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_SparseCtrlRGBPreprocessor',
            display_name='RGB SparseCtrl 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/preprocess',
            inputs=[
                io.Image.Input('image'),
                io.Vae.Input('vae'),
                io.Latent.Input('latent_size')
            ],
            outputs=[
                io.Image.Output('proc_IMAGE', is_output_list=False)
            ]
        )

    @classmethod
    def execute(cls, vae: VAE, image: Tensor, latent_size: Tensor):
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
        return io.NodeOutput(PreprocSparseRGBWrapper(condhint=encoded),)

class SparseWeightExtras(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_SparseCtrlWeightExtras',
            display_name='SparseCtrl Weight Extras 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/extras',
            inputs=[
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True),
                io.Float.Input('sparse_hint_mult', optional=True, default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('sparse_nonhint_mult', optional=True, default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('sparse_mask_mult', optional=True, default=1.0, max=10.0, min=0.0, step=0.001)
            ],
            outputs=[
                io.Custom('CN_WEIGHTS_EXTRAS').Output('cn_extras', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, cn_extras: dict[str]={}, sparse_hint_mult=1.0, sparse_nonhint_mult=1.0, sparse_mask_mult=1.0):
        cn_extras = cn_extras.copy()
        cn_extras[SparseConst.HINT_MULT] = sparse_hint_mult
        cn_extras[SparseConst.NONHINT_MULT] = sparse_nonhint_mult
        cn_extras[SparseConst.MASK_MULT] = sparse_mask_mult
        return io.NodeOutput(cn_extras, )
