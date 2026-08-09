import unittest
from types import SimpleNamespace
from unittest import mock

import transformers

from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
    _attach_comfy_h3_te_pre_quant_scales,
    _is_int8_embedding_checkpoint_weight,
)


class TestTextEncoderClassResolution(unittest.TestCase):
    """load_native must not load encoder-decoder text encoders via AutoModel.

    AutoModel maps T5/UMT5 model types to the full seq2seq class
    (T5Model/UMT5Model), whose forward needs decoder inputs and raises when the
    module is used purely as a text encoder.
    """

    server_args = SimpleNamespace(trust_remote_code=False, revision=None)

    def _resolve(self, is_encoder_decoder, architectures):
        config = SimpleNamespace(
            is_encoder_decoder=is_encoder_decoder, architectures=architectures
        )
        with mock.patch.object(
            transformers.AutoConfig, "from_pretrained", return_value=config
        ):
            return TextEncoderLoader._resolve_transformers_text_encoder_class(
                "dummy/path", self.server_args
            )

    def test_umt5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["UMT5EncoderModel"]), transformers.UMT5EncoderModel
        )
        self.assertIs(self._resolve(True, ["UMT5Model"]), transformers.UMT5EncoderModel)
        self.assertIs(
            self._resolve(True, ["UMT5ForConditionalGeneration"]),
            transformers.UMT5EncoderModel,
        )

    def test_t5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["T5EncoderModel"]), transformers.T5EncoderModel
        )
        self.assertIs(self._resolve(True, ["T5Model"]), transformers.T5EncoderModel)
        self.assertIs(
            self._resolve(True, ["T5ForConditionalGeneration"]),
            transformers.T5EncoderModel,
        )

    def test_mt5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["MT5EncoderModel"]), transformers.MT5EncoderModel
        )
        self.assertIs(self._resolve(True, ["MT5Model"]), transformers.MT5EncoderModel)
        self.assertIs(
            self._resolve(True, ["MT5ForConditionalGeneration"]),
            transformers.MT5EncoderModel,
        )

    def test_non_encoder_decoder_keeps_automodel(self):
        # e.g. CLIP/Mistral/Qwen text encoders are not encoder-decoder.
        self.assertIs(self._resolve(False, ["CLIPTextModel"]), transformers.AutoModel)

    def test_unknown_architecture_falls_back_to_automodel(self):
        self.assertIs(self._resolve(True, ["NotARealClass"]), transformers.AutoModel)

    def test_config_load_failure_falls_back_to_automodel(self):
        with mock.patch.object(
            transformers.AutoConfig,
            "from_pretrained",
            side_effect=OSError("no config"),
        ):
            cls = TextEncoderLoader._resolve_transformers_text_encoder_class(
                "dummy/path", self.server_args
            )
        self.assertIs(cls, transformers.AutoModel)


class TestMiniMaxH3CheckpointFilter(unittest.TestCase):
    def test_int8_embedding_weight_detection(self):
        import torch

        self.assertTrue(
            _is_int8_embedding_checkpoint_weight(
                "model.language_model.embed_tokens.weight",
                torch.empty(4, 8, dtype=torch.int8),
            )
        )
        self.assertFalse(
            _is_int8_embedding_checkpoint_weight(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                torch.empty(4, 8, dtype=torch.int8),
            )
        )

    def test_only_known_unconsumed_weights_are_filtered(self):
        should_load = MiniMaxH3Qwen3VLEncoder.should_materialize_checkpoint_weight
        expected = {
            "model.language_model.layers.49.self_attn.q_proj.weight": True,
            "model.language_model.layers.50.self_attn.q_proj.weight": False,
            "model.language_model.layers.63.mlp.down_proj.weight": False,
            "model.language_model.norm.weight": False,
            "lm_head.weight": False,
            "model.language_model.rotary_emb.inv_freq": False,
            "model.visual.blocks.0.attn.qkv.weight": True,
            "language_model.layers.63.mlp.down_proj.weight": True,
            "module.model.language_model.layers.63.mlp.down_proj.weight": True,
        }
        self.assertEqual(
            {name: should_load(name) for name in expected},
            expected,
        )

    def test_pre_quant_scale_buffers_attach_to_linears(self):
        import torch

        down = torch.nn.Linear(4, 2, bias=False)
        o_proj = torch.nn.Linear(4, 2, bias=False)
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = down
        layer.self_attn = torch.nn.Module()
        layer.self_attn.o_proj = o_proj
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.language_model = torch.nn.Module()
        model.model.language_model.layers = torch.nn.ModuleList([layer])
        pending = {
            "model.language_model.layers.0.mlp.down_proj.pre_quant_scale": torch.ones(
                4, dtype=torch.bfloat16
            ),
            "model.language_model.layers.0.self_attn.o_proj.pre_quant_scale": (
                torch.full((4,), 2.0, dtype=torch.bfloat16)
            ),
        }
        _attach_comfy_h3_te_pre_quant_scales(model, pending)
        self.assertTrue(hasattr(down, "pre_quant_scale"))
        self.assertTrue(hasattr(o_proj, "pre_quant_scale"))
        self.assertEqual(tuple(down.pre_quant_scale.shape), (4,))
        self.assertEqual(float(o_proj.pre_quant_scale[0]), 2.0)


if __name__ == "__main__":
    unittest.main()
