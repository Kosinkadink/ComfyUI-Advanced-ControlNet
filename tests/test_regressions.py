import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

comfyui_path = os.environ.get("COMFYUI_PATH")
if comfyui_path:
    sys.path.insert(0, comfyui_path)

import torch

from comfy.controlnet import T2IAdapter

from adv_control.control import T2IAdapterAdvanced
from adv_control.control_lllite import LLLiteModule
from adv_control.control_reference import REF_CONTROL_LIST_ALL, RefConst, refcn_diffusion_model_wrapper_factory


class LLLiteRegressionTests(unittest.TestCase):
    def create_module(self):
        torch.manual_seed(1)
        return LLLiteModule("test", False, 2, 1, 2, 2)

    def create_control(self, effect_mask=None, timestep_mask=None, uncond_multiplier=1.0):
        return SimpleNamespace(
            sub_idxs=None,
            cond_hint=torch.ones((1, 3, 8, 8)),
            latent_dims_div2=None,
            latent_dims_div4=None,
            mask_cond_hint=effect_mask,
            tk_mask_cond_hint=timestep_mask,
            latent_keyframes=None,
            weights=SimpleNamespace(
                has_uncond_multiplier=uncond_multiplier != 1.0,
                uncond_multiplier=uncond_multiplier,
            ),
            batched_number=2,
            cond_or_uncond=[0, 1],
            strength=1.0,
            _current_timestep_keyframe=SimpleNamespace(strength=1.0),
        )

    def test_unconditional_multiplier_uses_sampling_condition_order(self):
        control = self.create_control(uncond_multiplier=0.25)
        output = self.create_module()(torch.ones((2, 1, 2)), control)

        torch.testing.assert_close(output[1], output[0] * 0.25)

    def test_timestep_mask_applies_without_effect_mask(self):
        control = self.create_control(timestep_mask=torch.zeros((1, 8, 8)))
        output = self.create_module()(torch.ones((2, 1, 2)), control)

        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_effect_and_timestep_masks_are_combined(self):
        control = self.create_control(
            effect_mask=torch.ones((1, 8, 8)),
            timestep_mask=torch.zeros((1, 8, 8)),
        )
        output = self.create_module()(torch.ones((2, 1, 2)), control)

        torch.testing.assert_close(output, torch.zeros_like(output))


class T2IAdapterRegressionTests(unittest.TestCase):
    def test_sliding_context_extends_hint_to_full_latent_length(self):
        adapter = object.__new__(T2IAdapterAdvanced)
        adapter.sub_idxs = [2, 3]
        adapter.full_latent_length = 4
        adapter.cond_hint_original = torch.tensor([[[[7.0]]]])
        adapter.cond_hint = None
        adapter.prepare_mask_cond_hint = lambda **kwargs: None

        with patch.object(T2IAdapter, "get_control", lambda self, *args, **kwargs: self.cond_hint_original.clone()):
            output = adapter.get_control_advanced(torch.empty((2, 4, 1, 1)), None, None, 1, {})

        self.assertEqual(output.flatten().tolist(), [7.0, 7.0])
        self.assertEqual(adapter.cond_hint_original.flatten().tolist(), [7.0])


class ReferenceRegressionTests(unittest.TestCase):
    def test_cleanup_does_not_hide_original_exception(self):
        class ReferenceInjections:
            cleaned = False

            def clean_ref_module_mem(self):
                self.cleaned = True

        reference_injections = ReferenceInjections()
        wrapper = refcn_diffusion_model_wrapper_factory(reference_injections)
        transformer_options = {
            REF_CONTROL_LIST_ALL: [SimpleNamespace(should_run=lambda: True)],
            RefConst.REFCN_PRESENT_IN_CONDS: True,
        }

        with self.assertRaisesRegex(KeyError, "cond_or_uncond"):
            wrapper(lambda *args, **kwargs: None, torch.zeros(1), None, None, None, None, transformer_options)

        self.assertTrue(reference_injections.cleaned)


if __name__ == "__main__":
    unittest.main()
