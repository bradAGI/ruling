"""ModernBERT in MLX: a bidirectional encoder, not a generative model.

A port of Hugging Face's ModernBertForMaskedLM that takes explicit position
ids and a full boolean attention mask, so a caller can pack several
independent segments into one sequence. Every third layer attends globally;
the others see tokens within half of `local_attention` positions. Weights load
from the checkpoint's safetensors; the input embedding is tied to the decoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


@dataclass(frozen=True)
class Config:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    global_attn_every_n_layers: int
    local_attention: int
    global_rope_theta: float
    local_rope_theta: float
    norm_eps: float
    max_position_embeddings: int

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})

    def is_global(self, layer: int) -> bool:
        return layer % self.global_attn_every_n_layers == 0


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class Attention(nn.Module):
    def __init__(self, config: Config, theta: float):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self._inv_freq = 1.0 / (theta ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim))

    def __call__(self, x: mx.array, positions: mx.array, mask: mx.array) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.Wqkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        freqs = positions[:, :, None].astype(mx.float32) * self._inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)[:, None]
        cos, sin = mx.cos(emb).astype(q.dtype), mx.sin(emb).astype(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=mask[:, None])
        return self.Wo(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        value, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(value) * gate)


class Layer(nn.Module):
    def __init__(self, config: Config, index: int):
        super().__init__()
        self.is_global = config.is_global(index)
        self.window = config.local_attention // 2
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False) if index else None
        self.attn = Attention(config, config.global_rope_theta if self.is_global else config.local_rope_theta)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, positions: mx.array, full: mx.array, local: mx.array) -> mx.array:
        normed = self.attn_norm(x) if self.attn_norm is not None else x
        x = x + self.attn(normed, positions, full if self.is_global else local)
        return x + self.mlp(self.mlp_norm(x))


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.layers = [Layer(config, i) for i in range(config.num_hidden_layers)]
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)

    def __call__(self, tokens: mx.array, positions: mx.array, mask: mx.array) -> mx.array:
        """Hidden states [batch, length, hidden]. `mask[b, i, j]` is True where token i may attend to token j."""
        distance = mx.abs(positions[:, :, None] - positions[:, None, :])
        local = mask & (distance <= self.config.local_attention // 2)
        x = self.embed_norm(self.tok_embeddings(tokens))
        for layer in self.layers:
            x = layer(x, positions, mask, local)
        return self.final_norm(x)


class MaskedLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = Encoder(config)
        self.head_dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.head_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.decoder_bias = mx.zeros((config.vocab_size,))

    def transform(self, hidden: mx.array) -> mx.array:
        """The pretrained prediction transform applied before the vocabulary projection."""
        return self.head_norm(nn.gelu(self.head_dense(hidden)))

    def token_logits(self, hidden: mx.array) -> mx.array:
        return self.encoder.tok_embeddings.as_linear(self.transform(hidden)) + self.decoder_bias


RENAMES = {
    "decoder.weight": "encoder.tok_embeddings.weight",
    "decoder.bias": "decoder_bias",
    "head.dense.weight": "head_dense.weight",
    "head.norm.weight": "head_norm.weight",
    "model.embeddings.norm.weight": "encoder.embed_norm.weight",
    "model.final_norm.weight": "encoder.final_norm.weight",
}


def _rename(key: str) -> str:
    return RENAMES.get(key) or key.replace("model.layers.", "encoder.layers.")


def load(repo: str, revision: str) -> tuple[MaskedLM, AutoTokenizer, Config]:
    config = Config.from_dict(json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text()))
    model = MaskedLM(config)
    weights = mx.load(hf_hub_download(repo, "model.safetensors", revision=revision))
    model.load_weights([(_rename(k), v) for k, v in weights.items()], strict=True)
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    return model, tokenizer, config


def sequence_inputs(token_lists: list[list[int]], pad_id: int) -> tuple[mx.array, mx.array, mx.array]:
    """Plain batched inputs: positions from zero and a padding mask, as the reference model uses."""
    width = max(len(t) for t in token_lists)
    tokens = mx.array([t + [pad_id] * (width - len(t)) for t in token_lists])
    positions = mx.broadcast_to(mx.arange(width)[None], (len(token_lists), width))
    valid = mx.array([[i < len(t) for i in range(width)] for t in token_lists])
    return tokens, positions, valid[:, None, :] & valid[:, :, None]
