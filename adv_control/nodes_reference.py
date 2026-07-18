from comfy_api.latest import io
from torch import Tensor

from nodes import VAEEncode
import comfy.utils
from comfy.sd import VAE

from .control_reference import ReferenceAdvanced, ReferenceOptions, ReferenceType, ReferencePreprocWrapper

# node for ReferenceCN
class ReferenceControlNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_ReferenceControlNet',
            display_name='Reference ControlNet 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/Reference',
            inputs=[
                io.Combo.Input('reference_type', options=['reference_attn', 'reference_adain', 'reference_attn+adain']),
                io.Float.Input('style_fidelity', default=0.5, max=1.0, min=0.0, step=0.01),
                io.Float.Input('ref_weight', default=1.0, max=1.0, min=0.0, step=0.01)
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, reference_type: str, style_fidelity: float, ref_weight: float):
        ref_opts = ReferenceOptions.create_combo(reference_type=reference_type, style_fidelity=style_fidelity, ref_weight=ref_weight)
        controlnet = ReferenceAdvanced(ref_opts=ref_opts, timestep_keyframes=None)
        return io.NodeOutput(controlnet,)

class ReferenceControlFinetune(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_ReferenceControlNetFinetune',
            display_name='Reference ControlNet (Finetune) 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/Reference',
            inputs=[
                io.Float.Input('attn_style_fidelity', default=0.5, max=1.0, min=0.0, step=0.01),
                io.Float.Input('attn_ref_weight', default=1.0, max=1.0, min=0.0, step=0.01),
                io.Float.Input('attn_strength', default=1.0, max=1.0, min=0.0, step=0.01),
                io.Float.Input('adain_style_fidelity', default=0.5, max=1.0, min=0.0, step=0.01),
                io.Float.Input('adain_ref_weight', default=1.0, max=1.0, min=0.0, step=0.01),
                io.Float.Input('adain_strength', default=1.0, max=1.0, min=0.0, step=0.01)
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls,
                        attn_style_fidelity: float, attn_ref_weight: float, attn_strength: float,
                        adain_style_fidelity: float, adain_ref_weight: float, adain_strength: float):
        ref_opts = ReferenceOptions(reference_type=ReferenceType.ATTN_ADAIN,
                                    attn_style_fidelity=attn_style_fidelity, attn_ref_weight=attn_ref_weight, attn_strength=attn_strength,
                                    adain_style_fidelity=adain_style_fidelity, adain_ref_weight=adain_ref_weight, adain_strength=adain_strength)
        controlnet = ReferenceAdvanced(ref_opts=ref_opts, timestep_keyframes=None)
        return io.NodeOutput(controlnet,)

class ReferencePreprocessorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_ReferencePreprocessor',
            display_name='Reference Preproccessor 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/Reference/preprocess',
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
        return io.NodeOutput(ReferencePreprocWrapper(condhint=encoded),)
