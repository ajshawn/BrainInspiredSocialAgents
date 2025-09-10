"""
SimpleTransformerCore: Haiku RNNCore-compatible Transformer
----------------------------------------------------------

API mirrors the referenced IMPALANetwork:

class SimpleTransformerCore(hk.RNNCore):
  def __init__(self, num_actions, model_dim, num_heads, mlp_hidden_dim,
               num_layers, max_context_len, feature_extractor=None, dropout_rate=0.0): ...
  def __call__(self, inputs, state: ContextState): -> ((logits, value, op, emb), new_state)
  def initial_state(self, batch_size: int, **unused_kwargs) -> ContextState
  def unroll(self, inputs, state: ContextState) -> ((logits, value, op, emb), new_states)
  def critic(self, inputs): -> value

Notes:
- Receives one observation at a time in __call__; maintains a sliding window context inside the RNN state.
- No input embedding inside by default; pass a feature_extractor or leave None to use identity.
- Uses absolute sinusoidal positional embeddings; causal self-attention over [context || current].
- Plain residual connections (no GRU gating, no TXL memory).
- Shapes: inputs [B, D]; unroll inputs [T, B, D].
"""
from typing import Optional, Tuple, NamedTuple

from acme.specs import EnvironmentSpec
import haiku as hk
import jax.numpy as jnp
import jax
import numpy as np
from marl.agents.networks import make_haiku_networks_2

class ContextState(NamedTuple):
    buffer: jnp.ndarray  # [C,B,D]
    hidden: jnp.ndarray # For compatibility issue
    cell: jnp.ndarray  # For compatibility issue

def make_network_simple_transformer(
    environment_spec: EnvironmentSpec,
    feature_extractor: hk.Module,
    model_dim: int = 74,
    num_heads: int = 2,
    mlp_hidden_dim: int = 128,
    num_layers: int = 1,
    max_context_len: int = 20,
    dropout_rate: float = 0.0,
):
    def forward_fn(inputs, states: ContextState):
        core = SimpleTransformerCore(
            num_actions=environment_spec.actions.num_values,
            model_dim=model_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            max_context_len=max_context_len,
            feature_extractor=feature_extractor,
            dropout_rate=dropout_rate,
        )
        return core(inputs, states)
    
    def initial_state_fn(batch_size=None) -> ContextState:
        core = SimpleTransformerCore(
            num_actions=environment_spec.actions.num_values,
            model_dim=model_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            max_context_len=max_context_len,
            feature_extractor=feature_extractor,
            dropout_rate=dropout_rate,
        )
        return core.initial_state(batch_size=batch_size or 1)
    
    def unroll_fn(inputs, states: ContextState):
        core = SimpleTransformerCore(
            num_actions=environment_spec.actions.num_values,
            model_dim=model_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            max_context_len=max_context_len,
            feature_extractor=feature_extractor,
            dropout_rate=dropout_rate,
        )
        return core.unroll(inputs, states)
    
    def critic_fn(inputs):
        core = SimpleTransformerCore(
            num_actions=environment_spec.actions.num_values,
            model_dim=model_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            max_context_len=max_context_len,
            feature_extractor=feature_extractor,
            dropout_rate=dropout_rate,
        )
        return core.critic(inputs)
    
    return make_haiku_networks_2(
        env_spec=environment_spec,
        forward_fn=forward_fn,
        initial_state_fn=initial_state_fn,
        unroll_fn=unroll_fn,
        critic_fn=critic_fn,
    )

# -----------------------------
# Absolute Positional Embedding
# -----------------------------
class AbsPositionalEmbedding(hk.Module):
    def __init__(self, dim: int, name: Optional[str] = None):
        super().__init__(name=name)
        assert dim % 2 == 0, "Model dim must be even for sin/cos"
        self.dim = dim

    def __call__(self, length: int) -> jnp.ndarray:
        d = self.dim
        positions = jnp.arange(length, dtype=jnp.float32)[:, None]
        div = jnp.exp(jnp.arange(0, d, 2, dtype=jnp.float32) * (-jnp.log(10000.0) / d))
        pe = jnp.concatenate([jnp.sin(positions * div), jnp.cos(positions * div)], axis=1)
        return pe[:, None, :]  # [S,1,D]

# -----------------------------
# MHA / FFN / Encoder Layer
# -----------------------------
class MultiHeadAttention(hk.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout_rate: float = 0.0, name: Optional[str] = None):
        super().__init__(name=name)
        assert model_dim % num_heads == 0, "model_dim must divide num_heads"
        self.D = model_dim
        self.H = num_heads
        self.Hd = model_dim // num_heads
        self.dropout_rate = dropout_rate
        self.wq = hk.Linear(self.D, name="wq")
        self.wk = hk.Linear(self.D, name="wk")
        self.wv = hk.Linear(self.D, name="wv")
        self.wo = hk.Linear(self.D, name="wo")

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray], *, is_training: bool) -> jnp.ndarray:
        S, B, _ = x.shape
        H, Hd = self.H, self.Hd
        scale = 1.0 / jnp.sqrt(Hd)
        q = self.wq(x).reshape(S, B, H, Hd)
        k = self.wk(x).reshape(S, B, H, Hd)
        v = self.wv(x).reshape(S, B, H, Hd)
        scores = jnp.einsum('tbhm,sbhm->bhts', q, k) * scale  # [B,H,S,S]
        if mask is not None:
            scores = jnp.where(mask[None, None, :, :], jnp.array(-1e30, scores.dtype), scores)
        attn = jax.nn.softmax(scores, axis=-1)
        if is_training and self.dropout_rate > 0:
            attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, attn)
        out_heads = jnp.einsum('bhts,sbhm->bhtm', attn, v)  # [B,H,S,Hd]
        out = out_heads.transpose(2, 0, 1, 3) # [S B H Hd]
        out = out.reshape(S, B, -1) 
        out = self.wo(out)
        if is_training and self.dropout_rate > 0:
            out = hk.dropout(hk.next_rng_key(), self.dropout_rate, out)
        return out

class FeedForward(hk.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout_rate: float = 0.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.l1 = hk.Linear(hidden_dim, name="l1")
        self.l2 = hk.Linear(model_dim, name="l2")
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        h = jax.nn.relu(self.l1(x))
        if is_training and self.dropout_rate > 0:
            h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)
        h = self.l2(h)
        if is_training and self.dropout_rate > 0:
            h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)
        return h

class TransformerEncoderLayer(hk.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_hidden_dim: int, dropout_rate: float = 0.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.mha = MultiHeadAttention(model_dim, num_heads, dropout_rate, name="mha")
        self.ffn = FeedForward(model_dim, mlp_hidden_dim, dropout_rate, name="ffn")
        self.ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln1")
        self.ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln2")

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray], *, is_training: bool) -> jnp.ndarray:
        h = self.ln1(x)
        x = x + self.mha(h, mask, is_training=is_training)
        h = self.ln2(x)
        x = x + self.ffn(h, is_training=is_training)
        return x

# -----------------------------
# RNNCore-compatible Transformer
# -----------------------------
class SimpleTransformerCore(hk.Module):
    def __init__(
        self,
        num_actions: int,
        model_dim: int,
        feature_extractor,
        input_dim: int = 74,
        num_heads: int = 2,
        mlp_hidden_dim: int = 256,
        num_layers: int = 1,
        max_context_len: int = 20,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "simple_transformer_core")
        self.num_actions = num_actions
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.max_context_len = max_context_len
        self.dropout_rate = dropout_rate
        self._embed = feature_extractor(num_actions)

        self.layers = [
            TransformerEncoderLayer(model_dim, num_heads, mlp_hidden_dim, dropout_rate, name=f"enc_{i}")
            for i in range(num_layers)
        ]
        self.pos_emb = AbsPositionalEmbedding(input_dim, name="pos_emb")
        self._policy_layer = hk.Linear(num_actions, name="policy")
        self._value_layer = hk.Linear(1, name="value_layer")

    def initial_state(self, batch_size: int, **kwargs) -> ContextState:
        buf = jnp.zeros((self.max_context_len, batch_size, self.input_dim), dtype=jnp.float32)
        return ContextState(buffer=buf, hidden=jnp.array([0]), cell=jnp.array([0]))

    @staticmethod
    def _concat_and_crop(context: jnp.ndarray, x: jnp.ndarray, max_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # context: [C,B,D], x: [1,B,D]
        if x.ndim == 2: # [B, D]
            x = jnp.expand_dims(x, axis=0) # [1, B, D]
        full = jnp.concatenate([context, x], axis=0)
        start = np.maximum(0, full.shape[0] - max_len)
        return full, full[start:]

    def _encode_window(self, window: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        # window: [S,B,D]
        pe = self.pos_emb(window.shape[0])  # [S,1,D]
        h = window + pe
        if is_training and self.dropout_rate > 0:
            h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)
        for layer in self.layers:
            h = layer(h, mask=None, is_training=is_training)
        return h  # [S,B,D]

    def __call__(self, inputs, state: ContextState):
        # inputs expected [B, D_in]; run feature extractor if provided
        emb = self._embed(inputs)
        x = jnp.expand_dims(emb, axis=0)  # [T=1,B=1,D]
        full, new_buf = self._concat_and_crop(state.buffer, x, self.max_context_len)  # [S,B,D], [C,B,D]
        enc = self._encode_window(full, is_training=True)
        op = enc[-1]  # [B=1,D] representation for current step

        logits = self._policy_layer(op)             # [B=1, num_actions]
        logits = logits.reshape((self.num_actions,))
        value = jnp.squeeze(self._value_layer(op), axis=-1)  # [B]
        new_state = ContextState(buffer=new_buf, hidden=jnp.array([0]), cell=jnp.array([0]))
        return (logits, value, op, emb), new_state

    def unroll(self, inputs, state: ContextState):
        """Unroll over time dimension of inputs: [T,B,D_in] or feature-extracted directly."""
        emb_seq = self._embed(inputs)  # [T,B,D]

        def step(x_t, st):
            emb = x_t
            x = jnp.expand_dims(emb, axis=0)  # [1,B,D]
            buffer = getattr(st, "buffer", self.initial_state(batch_size=1).buffer)
            full, new_buf = self._concat_and_crop(buffer, x, self.max_context_len)  # [S,B,D], [C,B,D]
            enc = self._encode_window(full, is_training=True)
            op = enc[-1]  # [B,D] representation for current step
            new_state = ContextState(buffer=new_buf, hidden=jnp.array([0]), cell=jnp.array([0]))

            return op, new_state

        op, new_state = hk.static_unroll(step, emb_seq, state)
        logits = self._policy_layer(op)
        value = jnp.squeeze(self._value_layer(op), axis=-1)
        return (logits, value, op, emb_seq), new_state # logits [T,B,A], value [T,B], op [T,B,D], emb [T,B,D]

    def critic(self, inputs):
        return jnp.squeeze(self._value_layer(inputs), axis=-1)
