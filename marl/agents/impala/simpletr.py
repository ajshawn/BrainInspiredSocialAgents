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
from einops import rearrange
from marl.agents.networks import make_haiku_networks_2
from typing import Optional, List

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

def make_network_transformer_attention(
    environment_spec: EnvironmentSpec,
    feature_extractor: hk.Module,
    model_dim: int = 74,
    num_heads: int = 2,
    mlp_hidden_dim: int = 128,
    num_layers: int = 1,
    max_context_len: int = 20,
    dropout_rate: float = 0.0,
    num_heads_vis: int =2,
    key_size: int =64,
    positional_embedding: str =None, 
    add_selection_vec: bool = False, 
    attn_enhance_multiplier: float =0.0,
):
    def forward_fn(inputs, states: ContextState,mrng=None):
        core = SimpleTransformer_attention(
            num_actions=environment_spec.actions.num_values,
            model_dim=model_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            max_context_len=max_context_len,
            feature_extractor=feature_extractor,
            dropout_rate=dropout_rate,
            num_heads_vis=num_heads_vis,
            key_size=key_size,
            positional_embedding=positional_embedding, 
            add_selection_vec=add_selection_vec, 
            attn_enhance_multiplier=attn_enhance_multiplier,
        )
        return core(inputs, states, mrng=mrng)
    
    def initial_state_fn(batch_size=None) -> ContextState:
        core = SimpleTransformer_attention(
            num_actions=environment_spec.actions.num_values,
            model_dim=model_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            max_context_len=max_context_len,
            feature_extractor=feature_extractor,
            dropout_rate=dropout_rate,
            num_heads_vis=num_heads_vis,
            key_size=key_size,
            positional_embedding=positional_embedding, 
            add_selection_vec=add_selection_vec, 
            attn_enhance_multiplier=attn_enhance_multiplier,
        )
        return core.initial_state(batch_size=batch_size or 1)
    
    def unroll_fn(inputs, states: ContextState, mrng = None):
        core = SimpleTransformer_attention(
            num_actions=environment_spec.actions.num_values,
            model_dim=model_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            max_context_len=max_context_len,
            feature_extractor=feature_extractor,
            dropout_rate=dropout_rate,
            num_heads_vis=num_heads_vis,
            key_size=key_size,
            positional_embedding=positional_embedding, 
            add_selection_vec=add_selection_vec, 
            attn_enhance_multiplier=attn_enhance_multiplier,
        )
        return core.unroll(inputs, states, mrng = mrng)
    
    def critic_fn(inputs):
        core = SimpleTransformer_attention(
            num_actions=environment_spec.actions.num_values,
            model_dim=model_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            max_context_len=max_context_len,
            feature_extractor=feature_extractor,
            dropout_rate=dropout_rate,
            num_heads_vis=num_heads_vis,
            key_size=key_size,
            positional_embedding=positional_embedding, 
            add_selection_vec=add_selection_vec, 
            attn_enhance_multiplier=attn_enhance_multiplier,
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
        mlp_hidden_dim: int = 128,
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
        hidden = jnp.zeros((batch_size, self.model_dim), dtype=jnp.float32)
        return ContextState(buffer=buf, hidden=hidden, cell=jnp.array([0]))

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
        new_state = ContextState(buffer=new_buf, hidden=state.hidden, cell=jnp.array([0]))
        return (logits, value, op, emb), new_state

    def unroll(self, inputs, state: ContextState):
        """Unroll over time dimension of inputs: [T,B,D_in] or feature-extracted directly."""
        # vmap in loss function removes batch dimension
        emb_seq = self._embed(inputs)  # [T,B,D]

        def step(x_t, st):
            emb = x_t
            x = jnp.expand_dims(emb, axis=0)  # [1,B,D]
            buffer = getattr(st, "buffer", self.initial_state(batch_size=1).buffer)
            full, new_buf = self._concat_and_crop(buffer, x, self.max_context_len)  # [S,B,D], [C,B,D]
            enc = self._encode_window(full, is_training=True)
            op = enc[-1]  # [B,D] representation for current step
            new_state = ContextState(buffer=new_buf, hidden=state.hidden, cell=jnp.array([0]))

            return op, new_state

        op, new_state = hk.static_unroll(step, emb_seq, state)
        op = jnp.squeeze(op, axis=-2) # Remove the sequence length dim [T B seq hidden_dim]
        logits = self._policy_layer(op) # [T B action_dim]
        value = self._value_layer(op) # [T B 1]
        value = jnp.squeeze(value, axis=-1) # [T B]
   
        return (logits, value, op, emb_seq), new_state # logits [T,B,A], value [T,B], op [T,B,D], emb [T,B,D]

    def critic(self, inputs):
        return jnp.squeeze(self._value_layer(inputs), axis=-1)


class MultiHeadAttentionLayer_1selfattention(hk.Module):
  def __init__(
      self,
      num_heads: int,
      key_size_per_head: int,
      positional_embedding='learnable',
      add_selection_vec=False,
      attn_enhance_multiplier: float = 0.0,
      attn_enhance_head_indices: List[int] = [],
      dropout_rate: float = 0.25,
      temperature: float = 1.0,
      is_training: bool = True,
  ):
    super().__init__(name="multihead_attention_layer")
    self.num_heads = num_heads
    self.key_size_per_head = key_size_per_head  # per head
    self.positional_embedding = positional_embedding
    self.attn_enhance_multiplier = attn_enhance_multiplier
    self.add_selection_vec = add_selection_vec
    self.model_dim = num_heads * key_size_per_head
    self.attn_enhance_head_indices = np.array(attn_enhance_head_indices, dtype=int)
    self.dropout_rate = dropout_rate
    self.is_training = is_training
    self.temperature = temperature

  def __call__(self, query, key, value, mrng = None):
    if jnp.ndim(query) == 1:
      query = jnp.expand_dims(query, axis=0)
    if jnp.ndim(key) == 2:
      key = jnp.expand_dims(key, axis=0)
      value = jnp.expand_dims(value, axis=0)

    B, N, _ = key.shape

    # Positional embedding
    if self.positional_embedding == 'fixed':
      H = 11
      y = jnp.linspace(-1.0, 1.0, H)
      x = jnp.linspace(-1.0, 1.0, H)
      yy, xx = jnp.meshgrid(y, x, indexing="ij")
      coords = jnp.stack([yy, xx], axis=-1).reshape([N, 2])
      coords = jnp.broadcast_to(coords[None, ...], [B, N, 2])
      key = jnp.concatenate([key, coords], axis=-1)
      value = jnp.concatenate([value, coords], axis=-1)
    elif self.positional_embedding == 'learnable':
      pos_emb = hk.get_parameter("pos_embedding", shape=[N, self.model_dim], init=hk.initializers.TruncatedNormal(stddev=0.02))
      key += pos_emb[None, :, :]
      value += pos_emb[None, :, :]
    elif self.positional_embedding == 'frequency':
      # Linear projections
      # NOTE: Frequency positional embedding now only works for single head due to the concatenation
      
      H = W = 11  # assume inputs arranged as HxW grid

      # Create spatial coordinates
      y = jnp.linspace(0, 1, H)
      x = jnp.linspace(0, 1, W)
      yy, xx = jnp.meshgrid(y, x, indexing="ij")  # [H, W]

      # Define frequency basis size
      U = V = 8
      u = jnp.arange(1, U + 1)[None, None, :]  # [1,1,U]
      v = jnp.arange(1, V + 1)[None, None, :]  # [1,1,V]

      a = yy[:, :, None] * u * jnp.pi  # [H,W,U]
      b = xx[:, :, None] * v * jnp.pi  # [H,W,V]

      # Outer product of cosines across frequencies
      freq_basis = jnp.einsum("hwu,hwv->hwuv", jnp.cos(a), jnp.cos(b))
      freq_basis = freq_basis.reshape(H, W, -1)  # [H,W,U*V]

      # Tile for batch
      freq_basis = jnp.broadcast_to(freq_basis[None, ...], (B, H, W, U * V))
      freq_basis = freq_basis.reshape(B, N, -1)  # [B,N,U*V]

      # # Add frequency basis to projected keys and values
      key += freq_basis
      value += freq_basis
      # Concat frequency basis to projected keys and values
      # key = jnp.concatenate([key, freq_basis], axis=-1)  # [B, N, model_dim + U*V]
      # value = jnp.concatenate([value, freq_basis], axis=-1)  # [B, N, model_dim + U*V]
      
    q_proj_cross = hk.Linear(self.model_dim)(query)   # [B, model_dim]
    k_proj_cross = hk.Linear(self.model_dim)(key)     # [B, N, model_dim]
    v_proj_cross = hk.Linear(self.model_dim)(value)   # [B, N, model_dim]

    # --- Self-attention projections (all from key) ---
    q_proj_self = hk.Linear(self.key_size_per_head)(key)   # [B, N, K]
    k_proj_self = hk.Linear(self.key_size_per_head)(key)   # [B, N, K]
    v_proj_self = hk.Linear(self.key_size_per_head)(key)   # [B, N, K]

    grid_dim = k_proj_cross.shape[-2]

    # ----- Cross-attention heads (H-1 heads) -----
    q_cross = q_proj_cross.reshape(-1, self.num_heads, self.key_size_per_head)[:, 1:, :]  # [B, H-1, K]
    q_cross = q_cross[:, :, None, :]  # [B, H-1, 1, K]

    k_cross = k_proj_cross.reshape(-1, grid_dim, self.num_heads, self.key_size_per_head)[:, :, 1:, :]
    k_cross = jnp.transpose(k_cross, (0, 2, 1, 3))  # [B, H-1, N, K]

    v_cross = v_proj_cross.reshape(-1, grid_dim, self.num_heads, self.key_size_per_head)[:, :, 1:, :]
    v_cross = jnp.transpose(v_cross, (0, 2, 1, 3))  # [B, H-1, N, K]

    # ----- Self-attention head (head 0) -----
    q_self = q_proj_self.mean(axis=1)  # [B, K]
    q_self = q_self[:, None, None, :]  # [B, 1, 1, K]

    k_self = k_proj_self.reshape(-1, grid_dim, 1, self.key_size_per_head)
    k_self = jnp.transpose(k_self, (0, 2, 1, 3))  # [B, 1, N, K]

    v_self = v_proj_self.reshape(-1, grid_dim, 1, self.key_size_per_head)
    v_self = jnp.transpose(v_self, (0, 2, 1, 3))  # [B, 1, N, K]

    # ----- Concatenate into full multi-head tensors -----

    q = jnp.concatenate([q_self, q_cross], axis=1)  # [B, H, 1, K]
    k = jnp.concatenate([k_self, k_cross], axis=1)  # [B, H, N, K]
    v = jnp.concatenate([v_self, v_cross], axis=1)  # [B, H, N, K]

    # Scaled dot-product attention
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).squeeze(2)  # [B, H, N]

    scores = scores / jnp.sqrt(self.key_size_per_head)  # Scale scores
    weights = jax.nn.softmax(scores / self.temperature, axis=-1)  # [B, H, N]

    # Weighted sum
    output = jnp.einsum("bhn,bhnk->bhk", weights, v)  # [B, H, K]

    output = output.reshape(output.shape[0], -1)  # [B, model_dim]

    #breakpoint()
    if self.dropout_rate > 0.0 and self.is_training and mrng is not None:
        keep_prob = 1.0 - self.dropout_rate
        mask = jax.random.bernoulli(mrng, keep_prob, output.shape)
        mask = mask.astype(output.dtype)
        output = output * mask / keep_prob

    return output, weights



class SimpleTransformer_attention(SimpleTransformerCore):
    def __init__(
        self,
        num_actions: int,
        model_dim: int,
        feature_extractor,
        input_dim: int = 74,
        num_heads: int = 2,
        mlp_hidden_dim: int = 128,
        num_layers: int = 1,
        max_context_len: int = 20,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
        num_heads_vis=2,
        key_size=64,
        positional_embedding=None, 
        add_selection_vec=False, 
        attn_enhance_multiplier=0.0,
    ):
        super().__init__(
            num_actions=num_actions,
            model_dim=model_dim,
            feature_extractor=feature_extractor,
            input_dim=input_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            max_context_len=max_context_len,
            dropout_rate=dropout_rate,
            name=name or "simple_transformer_core")
        #from marl.agents.impala import MultiHeadAttentionLayer
        #self._attention = MultiHeadAttentionLayer(
        # num_heads=num_heads_vis,
        # key_size_per_head=key_size // num_heads_vis,
        # positional_embedding=positional_embedding,
        # add_selection_vec=add_selection_vec,
        # attn_enhance_multiplier=attn_enhance_multiplier)
        self._attention = MultiHeadAttentionLayer_1selfattention(
        num_heads=num_heads_vis,
        key_size_per_head=key_size // num_heads_vis,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier 
        )
        
    def __call__(self, inputs, state: ContextState, mrng=None):
        # inputs expected [B, D_in]; run feature extractor if provided
        embedding = self._embed(inputs) # [B, 121, F]
        # Apply attention between CNN features and LSTM hidden state
        attn_rng = None
        if mrng is not None:
            mrng, attn_rng = jax.random.split(mrng)

        attended, attn_weights = self._attention(state.hidden, embedding, embedding, mrng=attn_rng)
        obs = inputs["observation"]
        inventory, ready_to_shoot = obs["INVENTORY"], obs["READY_TO_SHOOT"]
        # Do a one-hot embedding of the actions.
        action = jax.nn.one_hot(
            inputs["action"], num_classes=self.num_actions)  # [B, A]
        # Add dummy trailing dimensions to rewards if necessary.
        while ready_to_shoot.ndim < inventory.ndim:
            ready_to_shoot = jnp.expand_dims(ready_to_shoot, axis=-1)
        if ready_to_shoot.ndim < attended.ndim:
            attended = jnp.squeeze(attended, axis=0)
        # Combine attention output and other observations
        combined = jnp.concatenate([attended, ready_to_shoot, inventory, action], axis=-1)

        x = jnp.expand_dims(combined, axis=0)  # [T=1,B=1,D]
        full, new_buf = self._concat_and_crop(state.buffer, x, self.max_context_len)  # [S,B,D], [C,B,D]
        enc = self._encode_window(full, is_training=True)
        op = enc[-1]  # [B=1,D] representation for current step
        logits = self._policy_layer(op)             # [B=1, num_actions]
        logits = logits.reshape((self.num_actions,))
        value = jnp.squeeze(self._value_layer(op), axis=-1)  # [B]
        new_state = ContextState(buffer=new_buf, hidden=op, cell=jnp.array([0]))
        return (logits, value, op, attn_weights), new_state

    def unroll(self, inputs, state: ContextState, mrng = None):
        """Unroll over time dimension of inputs: [T,B,D_in] or feature-extracted directly."""
        # vmap in loss function removes batch dimension
        op = self._embed(inputs) # [B, 121, F]
        flatten_op = jnp.reshape(op, (op.shape[0], -1))  # [B, ...]
        self.op_shape = op.shape  # save the original shape for later use
        self.flatten_op_dim = flatten_op.shape[-1]  # save the flattened dimension for later use
        # extract other observations
        obs = inputs["observation"]
        inventory, ready_to_shoot = obs["INVENTORY"], obs["READY_TO_SHOOT"]
        # Do a one-hot embedding of the actions.
        action = jax.nn.one_hot(
            inputs["action"], num_classes=self.num_actions)  # [B, A]
        # Add dummy trailing dimensions to rewards if necessary.
        while ready_to_shoot.ndim < inventory.ndim:
            ready_to_shoot = jnp.expand_dims(ready_to_shoot, axis=-1)
        # fix for dynamic_unroll with reverb sampled data
        # state = hk.LSTMState(state.hidden, state.cell)
        combined = jnp.concatenate(
            [flatten_op, ready_to_shoot, inventory, action], 
            axis=-1
        ) 
        
        T = inputs["action"].shape[0]
        step_rngs = None
        if mrng is not None:
            step_rngs = jax.random.split(mrng, T)  # one mrng per time step

        def step(input_t_tp, st):
            input_t, attn_rng = input_t_tp
            #breakpoint()
            # Slice out the flattened input for the current time step
            attn_input = input_t[:self.flatten_op_dim]
            # Reshape it back to the original shape
            attn_input = jnp.reshape(attn_input, self.op_shape[1:]) # skip batch/time dimension
            # Apply attention
            attended, attn_weights = self._attention(st.hidden, attn_input, attn_input, mrng = attn_rng)
            # Combine with the rest of the input
            rest_input = input_t[self.flatten_op_dim:]
            if rest_input.ndim < attended.ndim:
                attended = jnp.squeeze(attended, axis=0)
            combined_input = jnp.concatenate([attended, rest_input], axis=-1)
            x = jnp.expand_dims(combined_input, axis=0)  # [1,B,D]
            buffer = getattr(st, "buffer", self.initial_state(batch_size=1).buffer) 
            full, new_buf = self._concat_and_crop(buffer, x, self.max_context_len)  # [S,B,D], [C,B,D]
            enc = self._encode_window(full, is_training=True)
            op = enc[-1]  # [B,D] representation for current step
            new_state = ContextState(buffer=new_buf, hidden=op, cell=jnp.array([0]))

            return (op,attn_weights), new_state

        optuple, new_state = hk.static_unroll(step, (combined,step_rngs), state)
        op, attn_weights = optuple
        op = jnp.squeeze(op, axis=-2) # Remove the sequence length dim [T B seq hidden_dim]
        logits = self._policy_layer(op) # [T B action_dim]
        value = self._value_layer(op) # [T B 1]
        value = jnp.squeeze(value, axis=-1) # [T B]
   
        return (logits, value, op, attn_weights), new_state # logits [T,B,A], value [T,B], op [T,B,D], emb [T,B,D]

    