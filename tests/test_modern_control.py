import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

comfyui_path = os.environ.get("COMFYUI_PATH")
if comfyui_path:
    sys.path.insert(0, comfyui_path)

import torch

from comfy.controlnet import T2IAdapter

from adv_control.control import ControlNetAdvanced, T2IAdapterAdvanced
from adv_control.nodes_main import AdvancedControlNetApply, AdvancedControlNetInpaintingApply
from adv_control.utils import ControlWeights


class StopControlModel(Exception):
    pass


class ControlModel:
    dtype = torch.float32

    def __init__(self):
        self.hint = None

    def __call__(self, x, hint, timesteps, context, **kwargs):
        self.hint = hint
        raise StopControlModel


class VideoVAE:
    downscale_ratio = (4, 8, 8)

    def __init__(self):
        self.encoded_shape = None

    def spacial_compression_encode(self):
        return 8

    def encode(self, image):
        self.encoded_shape = image.shape
        return torch.ones((image.shape[0], 4, 2, 2, 2))


class ModernControlPreprocessingTests(unittest.TestCase):
    def test_effect_mask_is_resized_to_qwen_tokens(self):
        control = ControlNetAdvanced(ControlModel(), None)
        control.x_noisy_shape = (1, 16, 4, 6)
        control.mask_cond_hint = torch.tensor(
            [[[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 4]]
        )
        control.tk_mask_cond_hint = None
        control.weights = SimpleNamespace(has_uncond_multiplier=False, has_uncond_mask=False)
        control.latent_keyframes = None
        control._current_timestep_keyframe = SimpleNamespace(strength=1.0)

        output = torch.ones((1, 6, 4))
        control.apply_advanced_strengths_and_masks(output, batched_number=1)

        expected = torch.tensor(
            [[[0.0] * 4, [0.5] * 4, [1.0] * 4, [0.0] * 4, [0.5] * 4, [1.0] * 4]]
        )
        torch.testing.assert_close(output, expected)

    def test_effect_mask_matches_padded_flux_tokens_for_odd_latent_size(self):
        control = ControlNetAdvanced(ControlModel(), None)
        control.x_noisy_shape = (1, 16, 5, 7)
        control.mask_cond_hint = torch.ones((1, 1, 5, 7))
        control.tk_mask_cond_hint = None
        control.weights = SimpleNamespace(has_uncond_multiplier=False, has_uncond_mask=False)
        control.latent_keyframes = None
        control._current_timestep_keyframe = SimpleNamespace(strength=1.0)

        output = torch.ones((1, 12, 4))
        control.apply_advanced_strengths_and_masks(output, batched_number=1)

        torch.testing.assert_close(output, torch.ones_like(output))

    def test_vae_compression_and_source_mask_match_5d_hint(self):
        control_model = ControlModel()
        vae = VideoVAE()
        control = ControlNetAdvanced(control_model, None, compression_ratio=1, latent_format=SimpleNamespace(process_in=lambda value: value))
        control.real_compression_ratio = 1
        control.cond_hint_original = torch.ones((1, 3, 16, 16))
        control.cond_hint = None
        control.vae = vae
        control.extra_concat_orig = [torch.zeros((1, 1, 16, 16))]
        control.sub_idxs = None
        control.model_sampling_current = SimpleNamespace(timestep=lambda value: value, calculate_input=lambda timestep, value: value)
        control.prepare_mask_cond_hint = lambda **kwargs: None

        with self.assertRaises(StopControlModel):
            control.sliding_get_control(
                torch.ones((1, 4, 2, 2, 2)),
                torch.ones(1),
                {"c_crossattn": torch.ones((1, 1, 1))},
                1,
                {},
            )

        self.assertEqual(tuple(vae.encoded_shape), (1, 16, 16, 3))
        self.assertEqual(tuple(control_model.hint.shape), (1, 5, 2, 2, 2))


class T2IAdapterTests(unittest.TestCase):
    def test_effect_masks_are_applied_to_adapter_features(self):
        control = T2IAdapterAdvanced(SimpleNamespace(), None, channels_in=3)
        control.weights = ControlWeights.t2iadapter()
        control.latent_keyframes = None
        control.tk_mask_cond_hint = None
        control._current_timestep_keyframe = SimpleNamespace(strength=1.0)

        masks = {
            "zero": torch.zeros((1, 1, 8, 8)),
            "one": torch.ones((1, 1, 8, 8)),
            "half": torch.cat((torch.zeros((1, 1, 8, 4)), torch.ones((1, 1, 8, 4))), dim=3),
        }
        for name, mask in masks.items():
            with self.subTest(name=name):
                features = torch.ones((1, 4, 8, 8))
                control.mask_cond_hint = mask
                control.apply_advanced_strengths_and_masks(features, batched_number=1)
                torch.testing.assert_close(features, mask.expand_as(features))

    def test_sliding_context_extends_single_hint_to_full_latent_length(self):
        control = T2IAdapterAdvanced(SimpleNamespace(), None, channels_in=3)
        original_hint = torch.ones((1, 3, 8, 8))
        control.cond_hint_original = original_hint
        control.cond_hint = None
        control.sub_idxs = [2, 3]
        control.full_latent_length = 4
        control.prepare_mask_cond_hint = lambda **kwargs: None
        selected_hint = None

        def get_control(adapter, *args, **kwargs):
            nonlocal selected_hint
            selected_hint = adapter.cond_hint_original.clone()
            return sentinel.output

        with patch.object(T2IAdapter, "get_control", get_control):
            result = control.get_control_advanced(
                torch.ones((2, 4, 8, 8)),
                torch.ones(2),
                {},
                1,
                {},
            )

        self.assertIs(result, sentinel.output)
        self.assertEqual(tuple(selected_hint.shape), (2, 3, 8, 8))
        torch.testing.assert_close(selected_hint, original_hint.repeat(2, 1, 1, 1))
        self.assertIs(control.cond_hint_original, original_hint)


class AdvancedInpaintingApplyTests(unittest.TestCase):
    def test_source_mask_and_effect_mask_stay_independent(self):
        image = torch.ones((1, 2, 2, 3))
        inpaint_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        effect_mask = torch.full((1, 2, 2), 0.25)
        control_net = SimpleNamespace(concat_mask=True)

        with patch.object(AdvancedControlNetApply, "execute", return_value=sentinel.output) as apply:
            result = AdvancedControlNetInpaintingApply.execute(
                positive=sentinel.positive,
                negative=sentinel.negative,
                control_net=control_net,
                vae=sentinel.vae,
                image=image,
                inpaint_mask=inpaint_mask,
                strength=1.0,
                start_percent=0.0,
                end_percent=1.0,
                effect_mask_optional=effect_mask,
            )

        self.assertIs(result, sentinel.output)
        inputs = apply.call_args.kwargs
        torch.testing.assert_close(inputs["mask_optional"], effect_mask)
        torch.testing.assert_close(inputs["extra_concat"][0], 1.0 - inpaint_mask.unsqueeze(1))
        torch.testing.assert_close(
            inputs["image"],
            torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]]),
        )

    def test_non_inpaint_control_has_readable_error(self):
        with self.assertRaisesRegex(ValueError, "does not use an inpaint source mask"):
            AdvancedControlNetInpaintingApply.execute(
                positive=[],
                negative=[],
                control_net=SimpleNamespace(concat_mask=False),
                vae=sentinel.vae,
                image=torch.ones((1, 2, 2, 3)),
                inpaint_mask=torch.zeros((1, 2, 2)),
                strength=1.0,
                start_percent=0.0,
                end_percent=1.0,
            )


if __name__ == "__main__":
    unittest.main()
