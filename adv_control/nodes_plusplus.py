from comfy_api.latest import io
from torch import Tensor
import math

import folder_paths

from .control_plusplus import load_controlnetplusplus, PlusPlusInput, PlusPlusInputGroup, PlusPlusImageWrapper

class PlusPlusLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_ControlNet++LoaderAdvanced',
            display_name='Load ControlNet++ Model (Multi) 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/ControlNet++',
            inputs=[
                io.Custom('PLUS_INPUT').Input('plus_input'),
                io.Combo.Input('name', options=folder_paths.get_filename_list("controlnet"))
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False),
                io.Image.Output('IMAGE', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, plus_input: PlusPlusInputGroup, name: str):
        controlnet_path = folder_paths.get_full_path("controlnet", name)
        controlnet = load_controlnetplusplus(controlnet_path)
        controlnet.verify_control_type(name, plus_input)
        controlnet.allow_condhint_latents = True
        return io.NodeOutput(controlnet, PlusPlusImageWrapper(plus_input),)

class PlusPlusLoaderSingle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_ControlNet++LoaderSingle',
            display_name='Load ControlNet++ Model (Single) 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/ControlNet++',
            inputs=[
                io.Combo.Input('name', options=folder_paths.get_filename_list("controlnet")),
                io.Combo.Input('control_type', options=['openpose', 'depth', 'hed/pidi/scribble/ted', 'canny/lineart/mlsd', 'normal', 'segment', 'tile', 'inpaint/outpaint', 'none'], default='none')
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, name: str, control_type: str):
        controlnet_path = folder_paths.get_full_path("controlnet", name)
        controlnet = load_controlnetplusplus(controlnet_path)
        controlnet.single_control_type = control_type
        controlnet.verify_control_type(name)
        return io.NodeOutput(controlnet,)

class PlusPlusInputNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_ControlNet++InputNode',
            display_name='ControlNet++ Input 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/ControlNet++',
            inputs=[
                io.Image.Input('image'),
                io.Combo.Input('control_type', options=['openpose', 'depth', 'hed/pidi/scribble/ted', 'canny/lineart/mlsd', 'normal', 'segment', 'tile', 'inpaint/outpaint']),
                io.Custom('PLUS_INPUT').Input('prev_plus_input', optional=True)
            ],
            outputs=[
                io.Custom('PLUS_INPUT').Output('PLUS_INPUT', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, image: Tensor, control_type: str, strength=1.0, prev_plus_input: PlusPlusInputGroup=None):
        if prev_plus_input is None:
            prev_plus_input = PlusPlusInputGroup()
        prev_plus_input = prev_plus_input.clone()

        if math.isclose(strength, 0.0):
            strength = 0.0000001
        pp_input = PlusPlusInput(image, control_type, strength)
        prev_plus_input.add(pp_input)

        return io.NodeOutput(prev_plus_input,)
