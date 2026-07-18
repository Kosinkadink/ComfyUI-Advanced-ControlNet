from comfy_api.latest import io
import os

import torch
import folder_paths

import numpy as np
from PIL import Image, ImageOps
from .control import load_controlnet, is_advanced_controlnet
from .nodes_main import AdvancedControlNetApply
from .utils import ControlWeights, TimestepKeyframeGroup, TimestepKeyframe, get_properly_arranged_t2i_weights

class LoadImagesFromDirectory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='LoadImagesFromDirectory',
            display_name='🚫Load Images [DEPRECATED] 🛂🅐🅒🅝',
            category='',
            inputs=[
                io.String.Input('directory', default=''),
                io.Int.Input('image_load_cap', optional=True, default=0, max=9007199254740991, min=0, step=1),
                io.Int.Input('start_index', optional=True, default=0, max=9007199254740991, min=0, step=1)
            ],
            outputs=[
                io.Image.Output('IMAGE', is_output_list=False),
                io.Mask.Output('MASK', is_output_list=False),
                io.Int.Output('INT', is_output_list=False)
            ],
            is_deprecated=True
        )
    

    @classmethod
    def execute(cls, directory: str, image_load_cap: int = 0, start_index: int = 0):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]
        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_count += 1
        
        if len(images) == 0:
            raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

        return io.NodeOutput(torch.cat(images, dim=0), torch.stack(masks, dim=0), image_count)

class ScaledSoftUniversalWeightsDeprecated(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ScaledSoftControlNetWeights',
            display_name='Scaled Soft Weights 🛂🅐🅒🅝',
            category='',
            inputs=[
                io.Float.Input('base_multiplier', default=0.825, max=1.0, min=0.0, step=0.001),
                io.Boolean.Input('flip_weights', default=False),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, base_multiplier, flip_weights, uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = ControlWeights.universal(base_multiplier=base_multiplier, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class SoftControlNetWeightsDeprecated(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SoftControlNetWeights',
            display_name='ControlNet Soft Weights 🛂🅐🅒🅝',
            category='',
            inputs=[
                io.Float.Input('weight_00', default=0.09941396206337118, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_01', default=0.12050177219802567, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_02', default=0.14606275417942507, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_03', default=0.17704576264172736, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_04', default=0.214600924414215, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_05', default=0.26012233262329093, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_06', default=0.3152997971191405, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_07', default=0.3821815722656249, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_08', default=0.4632503906249999, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_09', default=0.561515625, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_10', default=0.6806249999999999, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_11', default=0.825, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_12', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Boolean.Input('flip_weights', default=False),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ],
            is_deprecated=True
        )
    

    @classmethod
    def execute(cls, weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06,
                     weight_07, weight_08, weight_09, weight_10, weight_11, weight_12, flip_weights,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights_output = [weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                   weight_07, weight_08, weight_09, weight_10, weight_11]
        weights_middle = [weight_12]
        weights = ControlWeights.controlnet(weights_output=weights_output, weights_middle=weights_middle, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class CustomControlNetWeightsDeprecated(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='CustomControlNetWeights',
            display_name='ControlNet Custom Weights 🛂🅐🅒🅝',
            category='',
            inputs=[
                io.Float.Input('weight_00', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_01', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_02', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_03', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_04', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_05', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_06', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_07', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_08', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_09', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_10', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_11', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_12', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Boolean.Input('flip_weights', default=False),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ],
            is_deprecated=True
        )
    

    @classmethod
    def execute(cls, weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06,
                     weight_07, weight_08, weight_09, weight_10, weight_11, weight_12, flip_weights,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights_output = [weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                   weight_07, weight_08, weight_09, weight_10, weight_11]
        weights_middle = [weight_12]
        weights = ControlWeights.controlnet(weights_output=weights_output, weights_middle=weights_middle, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class SoftT2IAdapterWeightsDeprecated(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SoftT2IAdapterWeights',
            display_name='T2IAdapter Soft Weights 🛂🅐🅒🅝',
            category='',
            inputs=[
                io.Float.Input('weight_00', default=0.25, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_01', default=0.62, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_02', default=0.825, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_03', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Boolean.Input('flip_weights', default=False),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ],
            is_deprecated=True
        )
    

    @classmethod
    def execute(cls, weight_00, weight_01, weight_02, weight_03, flip_weights,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = [weight_00, weight_01, weight_02, weight_03]
        weights = get_properly_arranged_t2i_weights(weights)
        weights = ControlWeights.t2iadapter(weights_input=weights, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class CustomT2IAdapterWeightsDeprecated(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='CustomT2IAdapterWeights',
            display_name='T2IAdapter Custom Weights 🛂🅐🅒🅝',
            category='',
            inputs=[
                io.Float.Input('weight_00', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_01', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_02', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Float.Input('weight_03', default=1.0, max=10.0, min=0.0, step=0.001),
                io.Boolean.Input('flip_weights', default=False),
                io.Float.Input('uncond_multiplier', optional=True, default=1.0, max=1.0, min=0.0, step=0.01),
                io.Custom('CN_WEIGHTS_EXTRAS').Input('cn_extras', optional=True)
            ],
            outputs=[
                io.Custom('CONTROL_NET_WEIGHTS').Output('CN_WEIGHTS', is_output_list=False),
                io.Custom('TIMESTEP_KEYFRAME').Output('TK_SHORTCUT', is_output_list=False)
            ],
            is_deprecated=True
        )
    

    @classmethod
    def execute(cls, weight_00, weight_01, weight_02, weight_03, flip_weights,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = [weight_00, weight_01, weight_02, weight_03]
        weights = get_properly_arranged_t2i_weights(weights)
        weights = ControlWeights.t2iadapter(weights_input=weights, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return io.NodeOutput(weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

class AdvancedControlNetApplyDEPR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_AdvancedControlNetApply',
            display_name='Apply Advanced ControlNet 🛂🅐🅒🅝',
            category='',
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
                io.Model.Input('model_optional', optional=True),
                io.Vae.Input('vae_optional', optional=True)
            ],
            outputs=[
                io.Conditioning.Output('positive', is_output_list=False),
                io.Conditioning.Output('negative', is_output_list=False),
                io.Model.Output('model_opt', is_output_list=False)
            ],
            is_deprecated=True
        )

    @classmethod
    def execute(cls, positive, negative, control_net, image, strength, start_percent, end_percent,
                         mask_optional=None, model_optional=None, vae_optional=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override=None,
                         weights_override: ControlWeights=None, control_apply_to_uncond=False):
        new_positive, new_negative = AdvancedControlNetApply.execute(positive=positive, negative=negative, control_net=control_net, image=image,
                                                          strength=strength, start_percent=start_percent, end_percent=end_percent,
                                                          mask_optional=mask_optional, vae_optional=vae_optional,
                                                          timestep_kf=timestep_kf, latent_kf_override=latent_kf_override, weights_override=weights_override,).args
        return io.NodeOutput(new_positive, new_negative, model_optional)

class AdvancedControlNetApplySingleDEPR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_AdvancedControlNetApplySingle',
            display_name='Apply Advanced ControlNet(1) 🛂🅐🅒🅝',
            category='',
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
                io.Model.Input('model_optional', optional=True),
                io.Vae.Input('vae_optional', optional=True)
            ],
            outputs=[
                io.Conditioning.Output('CONDITIONING', is_output_list=False),
                io.Model.Output('model_opt', is_output_list=False)
            ],
            is_deprecated=True
        )

    @classmethod
    def execute(cls, conditioning, control_net, image, strength, start_percent, end_percent,
                         mask_optional=None, model_optional=None, vae_optional=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override=None,
                         weights_override: ControlWeights=None):
        values = AdvancedControlNetApply.execute(positive=conditioning, negative=None, control_net=control_net, image=image,
                                                          strength=strength, start_percent=start_percent, end_percent=end_percent,
                                                          mask_optional=mask_optional, vae_optional=vae_optional,
                                                          timestep_kf=timestep_kf, latent_kf_override=latent_kf_override, weights_override=weights_override,
                                                          control_apply_to_uncond=True)
        return io.NodeOutput(values.args[0], model_optional)

class ControlNetLoaderAdvancedDEPR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ControlNetLoaderAdvanced',
            display_name='Load Advanced ControlNet Model 🛂🅐🅒🅝',
            category='',
            inputs=[
                io.Combo.Input('control_net_name', options=folder_paths.get_filename_list("controlnet")),
                io.Custom('TIMESTEP_KEYFRAME').Input('tk_optional', optional=True)
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ],
            is_deprecated=True
        )

    @classmethod
    def execute(cls, control_net_name,
                        tk_optional: TimestepKeyframeGroup=None,
                        timestep_keyframe: TimestepKeyframeGroup=None,
                        ):
        if timestep_keyframe is not None: # backwards compatibility
            tk_optional = timestep_keyframe
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, tk_optional)
        return io.NodeOutput(controlnet,)
    

class DiffControlNetLoaderAdvancedDEPR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='DiffControlNetLoaderAdvanced',
            display_name='Load Advanced ControlNet Model (diff) 🛂🅐🅒🅝',
            category='',
            inputs=[
                io.Model.Input('model'),
                io.Combo.Input('control_net_name', options=folder_paths.get_filename_list("controlnet")),
                io.Custom('TIMESTEP_KEYFRAME').Input('tk_optional', optional=True)
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ],
            is_deprecated=True
        )
    

    @classmethod
    def execute(cls, control_net_name, model,
                        tk_optional: TimestepKeyframeGroup=None,
                        timestep_keyframe: TimestepKeyframeGroup=None
                        ):
        if timestep_keyframe is not None: # backwards compatibility
            tk_optional = timestep_keyframe
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, tk_optional, model)
        if is_advanced_controlnet(controlnet):
            controlnet.verify_all_weights()
        return io.NodeOutput(controlnet,)
