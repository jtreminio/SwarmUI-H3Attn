from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "comfy_node"
    / "SwarmH3WindowAttention"
    / "window_attention.py"
)
SPEC = spec_from_file_location("swarm_h3_window_attention_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
window_attention = module_from_spec(SPEC)
sys.modules[SPEC.name] = window_attention
SPEC.loader.exec_module(window_attention)


class VideoBlockMetadataTests(unittest.TestCase):
    def assert_exact_policy(self, runtime, window_frames):
        partial_rows, full_rows = window_attention._video_block_rows(
            runtime, window_frames
        )
        radius = window_frames // 2
        video_rows = runtime.video_stop - runtime.video_start

        for query_index in range(video_rows):
            query_block = query_index // window_attention._MASK_BLOCK_SIZE
            query_frame = query_index // runtime.frame_rows
            partial = set(partial_rows[query_block])
            full = set(full_rows[query_block])
            self.assertTrue(partial.isdisjoint(full))
            for key_index in range(runtime.video_stop):
                key_block = key_index // window_attention._MASK_BLOCK_SIZE
                represented = key_block in full
                if key_block in partial:
                    represented = (
                        key_index < runtime.video_start
                        or abs(
                            query_frame
                            - (key_index - runtime.video_start) // runtime.frame_rows
                        )
                        <= radius
                    )
                expected = (
                    key_index < runtime.video_start
                    or abs(
                        query_frame
                        - (key_index - runtime.video_start) // runtime.frame_rows
                    )
                    <= radius
                )
                self.assertEqual(expected, represented)

    def test_unaligned_boundaries_match_token_policy(self):
        runtime = window_attention.WindowRuntime(
            video_start=139,
            video_stop=594,
            latent_frames=7,
            frame_rows=65,
        )
        for window_frames in (1, 3, 5, 9):
            with self.subTest(window_frames=window_frames):
                self.assert_exact_policy(runtime, window_frames)

        partial_rows, full_rows = window_attention._video_block_rows(runtime, 3)
        self.assertTrue(any(full_rows))
        boundary_block = runtime.video_start // window_attention._MASK_BLOCK_SIZE
        self.assertTrue(any(boundary_block in row for row in partial_rows))
        self.assertFalse(any(boundary_block in row for row in full_rows))

    def test_aligned_blocks_produce_full_temporal_interior(self):
        runtime = window_attention.WindowRuntime(
            video_start=256,
            video_stop=896,
            latent_frames=5,
            frame_rows=128,
        )
        self.assert_exact_policy(runtime, window_frames=3)
        partial_rows, full_rows = window_attention._video_block_rows(runtime, 3)
        self.assertTrue(all(full_rows))
        self.assertTrue(all(not row for row in partial_rows))

        mask = window_attention._video_mask_for(
            runtime,
            window_attention.WindowConfig(
                window_seconds=1.0,
                window_frames=3,
                dense_layers=frozenset(),
                transformer_layers=1,
                verbose=False,
            ),
            torch.device("cpu"),
        )
        self.assertEqual((640, 896), mask.seq_lengths)
        self.assertIsNone(mask.q_indices)
        self.assertEqual(mask.kv_indices.shape[-1], mask.full_kv_indices.shape[-1])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by the H3 override")
class MixedAttentionTests(unittest.TestCase):
    def test_full_blocks_match_dense_mask_across_query_blocks(self):
        runtime = window_attention.WindowRuntime(
            video_start=256,
            video_stop=896,
            latent_frames=5,
            frame_rows=128,
        )
        config = window_attention.WindowConfig(
            window_seconds=1.0,
            window_frames=3,
            dense_layers=frozenset(),
            transformer_layers=1,
            verbose=False,
        )
        torch.manual_seed(1234)
        query = torch.randn(
            (1, 1, 640, 128), device="cuda", dtype=torch.bfloat16
        )
        key = torch.randn(
            (1, 1, 896, 128), device="cuda", dtype=torch.bfloat16
        )
        value = torch.randn_like(key)
        block_mask = window_attention._video_mask_for(runtime, config, query.device)
        output = window_attention._compiled_flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            kernel_options={
                "BACKEND": "TRITON",
                "ROWS_GUARANTEED_SAFE": True,
                "BLOCKS_ARE_CONTIGUOUS": False,
            },
        )

        query_index = torch.arange(640, device="cuda")[:, None]
        key_index = torch.arange(896, device="cuda")[None, :]
        allowed = (key_index < 256) | (
            (key_index >= 256)
            & ((query_index // 128 - (key_index - 256) // 128).abs() <= 1)
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed
        )
        torch.cuda.synchronize()

        self.assertEqual(
            block_mask.kv_indices.shape[-1],
            block_mask.full_kv_indices.shape[-1],
        )
        self.assertLessEqual(float((output - reference).abs().max()), 0.002)

    def test_global_queries_use_configured_dense_backend(self):
        runtime = window_attention.WindowRuntime(
            video_start=128,
            video_stop=512,
            latent_frames=3,
            frame_rows=128,
        )
        config = window_attention.WindowConfig(
            window_seconds=1.0,
            window_frames=1,
            dense_layers=frozenset(),
            transformer_layers=1,
            verbose=False,
        )
        options = {window_attention._RUNTIME_KEY: runtime}
        q = torch.randn((1, 2, 512, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        calls = []

        def configured_dense(query, key, value, heads, **kwargs):
            calls.append((query.shape, key.shape, value.shape, heads, kwargs))
            return torch.full_like(query, 11)

        def fake_flex(query, key, value, **kwargs):
            self.assertEqual((1, 2, 384, 128), tuple(query.shape))
            self.assertEqual((1, 2, 512, 128), tuple(key.shape))
            return torch.full_like(query, 22)

        def previous(*_args, **_kwargs):
            raise AssertionError("hybrid global rows must bypass sparse overrides")

        override = window_attention._make_attention_override(config, previous)
        with (
            patch.object(window_attention, "_video_mask_for", return_value=object()),
            patch.object(window_attention, "_compiled_flex_attention", fake_flex),
        ):
            output = override(
                configured_dense,
                q,
                k,
                v,
                2,
                skip_reshape=True,
                transformer_options=options,
            )

        self.assertEqual((1, 512, 256), tuple(output.shape))
        self.assertEqual(1, len(calls))
        dense_q, dense_k, dense_v, heads, kwargs = calls[0]
        self.assertEqual((1, 2, 128, 128), tuple(dense_q))
        self.assertEqual((1, 2, 512, 128), tuple(dense_k))
        self.assertEqual(dense_k, dense_v)
        self.assertEqual(2, heads)
        self.assertTrue(kwargs["skip_reshape"])
        self.assertTrue(kwargs["skip_output_reshape"])
        self.assertTrue(torch.all(output[:, :128] == 11))
        self.assertTrue(torch.all(output[:, 128:] == 22))


if __name__ == "__main__":
    unittest.main()
