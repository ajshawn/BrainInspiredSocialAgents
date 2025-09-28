from acme.jax import utils
from acme.specs import EnvironmentSpec
import haiku as hk
import jax.numpy as jnp
import jax
import numpy as np
from marl.agents.networks import make_haiku_networks
from marl.agents.networks import make_haiku_networks_2
from marl.agents.impala.simpletr import ContextState

from typing import Optional, List

# Useful type aliases
Images = jnp.ndarray

batch_concat = utils.batch_concat
add_batch_dim = utils.add_batch_dim


def make_network(environment_spec: EnvironmentSpec,
                 feature_extractor: hk.Module,
                 recurrent_dim: int = 128):

  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.unroll(inputs, state)

  return make_haiku_networks(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
  )

def make_network_2(environment_spec: EnvironmentSpec,
                   feature_extractor: hk.Module,
                   recurrent_dim: int = 128,
                   ):

  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        )
    return model(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        )
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        )
    return model.unroll(inputs, state)

  def critic_fn(inputs):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        )
    return model.critic(inputs)

  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )

def make_network_attention(environment_spec: EnvironmentSpec,
                   feature_extractor: hk.Module,
                   recurrent_dim: int = 128,
                   positional_embedding: Optional[str] = None,
                   add_selection_vec: bool = False):

  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec)
    return model(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec)
    return model.unroll(inputs, state)

  def critic_fn(inputs):
    model = IMPALANetwork_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec)
    return model.critic(inputs)

  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )

def make_network_attention_tanh(environment_spec: EnvironmentSpec,
                   feature_extractor: hk.Module,
                   recurrent_dim: int = 128,
                   positional_embedding: bool = True):

  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention_tanh(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding)
    return model(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_attention_tanh(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention_tanh(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding)
    return model.unroll(inputs, state)

  def critic_fn(inputs):
    model = IMPALANetwork_attention_tanh(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding)
    return model.critic(inputs)

  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )
  
def make_network_attention_spatial(environment_spec: EnvironmentSpec,
                   feature_extractor: hk.Module,
                   recurrent_dim: int = 128,
                   add_selection_vec: bool = False):
  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention_spatial(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        add_selection_vec=add_selection_vec)
    return model(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_attention_spatial(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        add_selection_vec=add_selection_vec)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention_spatial(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        add_selection_vec=add_selection_vec)
    return model.unroll(inputs, state)

  def critic_fn(inputs):
    model = IMPALANetwork_attention_spatial(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        add_selection_vec=add_selection_vec)
    return model.critic(inputs)

  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )

def make_network_attention_item_aware(environment_spec: EnvironmentSpec,
                    feature_extractor: hk.Module,
                    recurrent_dim: int = 128,
                    positional_embedding: Optional[str] = None,
                    attn_enhance_multiplier: float = 1.0):
  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention_item_aware(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        attn_enhance_multiplier=attn_enhance_multiplier)
    return model(inputs, state)
  
  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_attention_item_aware(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        attn_enhance_multiplier=attn_enhance_multiplier)
    return model.initial_state(batch_size)
  
  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention_item_aware(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        attn_enhance_multiplier=attn_enhance_multiplier)
    return model.unroll(inputs, state)
  
  def critic_fn(inputs):
    model = IMPALANetwork_attention_item_aware(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        attn_enhance_multiplier=attn_enhance_multiplier)
    return model.critic(inputs)
  
  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )

def make_network_attention_multihead(environment_spec: EnvironmentSpec,
                                feature_extractor: hk.Module,
                                recurrent_dim: int = 128,
                                positional_embedding: Optional[str] = None,
                                add_selection_vec: bool = False,
                                attn_enhance_multiplier: float = 0.0,
                                num_heads: int = 4,
                                key_size: int = 64):   
  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size)
    return model(inputs, state) 

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_multihead_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size)
    return model.unroll(inputs, state)

  def critic_fn(inputs):
    model = IMPALANetwork_multihead_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size)
    return model.critic(inputs)

  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )                       

def make_network_attention_multihead_disturb(environment_spec: EnvironmentSpec,
                                feature_extractor: hk.Module,
                                recurrent_dim: int = 128,
                                positional_embedding: Optional[str] = None,
                                add_selection_vec: bool = False,
                                attn_enhance_multiplier: float = 0.0,
                                num_heads: int = 4,
                                key_size: int = 64,
                                disturb_heads: List[int] = []):   
  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention_disturb(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size,
        disturb_head_indices=disturb_heads)
    return model(inputs, state) 

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_multihead_attention_disturb(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size,
        disturb_head_indices=disturb_heads)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention_disturb(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size,
        disturb_head_indices=disturb_heads)
    return model.unroll(inputs, state)

  def critic_fn(inputs):
    model = IMPALANetwork_multihead_attention_disturb(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size,
        disturb_head_indices=disturb_heads)
    return model.critic(inputs)
  
  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )

def make_network_attention_multihead_enhance(environment_spec: EnvironmentSpec,
                                feature_extractor: hk.Module,
                                recurrent_dim: int = 128,
                                positional_embedding: Optional[str] = None,
                                add_selection_vec: bool = False,
                                attn_enhance_multiplier: float = 0.0,
                                num_heads: int = 4,
                                key_size: int = 64,
                                attn_enhance_head_indices: List[int] = [],
                                attn_enhance_item_idx: int = 0):
  """
  Create a multi-head attention network with enhanced attention on specified heads.
  """
  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention_enhance(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size,
        attn_enhance_head_indices=attn_enhance_head_indices,
        attn_enhance_item_idx=attn_enhance_item_idx)
    return model(inputs, state)
  
  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_multihead_attention_enhance(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size,
        attn_enhance_head_indices=attn_enhance_head_indices,
        attn_enhance_item_idx=attn_enhance_item_idx)
    return model.initial_state(batch_size)
  
  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention_enhance(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size,
        attn_enhance_head_indices=attn_enhance_head_indices,
        attn_enhance_item_idx=attn_enhance_item_idx)
    return model.unroll(inputs, state)
  
  def critic_fn(inputs):
    model = IMPALANetwork_multihead_attention_enhance(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        num_heads=num_heads,
        key_size=key_size,
        attn_enhance_head_indices=attn_enhance_head_indices,
        attn_enhance_item_idx=attn_enhance_item_idx)
    return model.critic(inputs)

  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )

def make_network_attention_multihead_item_aware(environment_spec: EnvironmentSpec,
                                feature_extractor: hk.Module,
                                recurrent_dim: int = 128,
                                positional_embedding: Optional[str] = None,
                                add_selection_vec: bool = False,                            
                                num_heads: int = 4,
                                key_size: int = 64):   
  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention_item_aware(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        num_heads=num_heads,
        key_size=key_size)
    return model(inputs, state) 

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_multihead_attention_item_aware(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        num_heads=num_heads,
        key_size=key_size)
    return model.initial_state(batch_size)
  
  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention_item_aware(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        num_heads=num_heads,
        key_size=key_size)
    return model.unroll(inputs, state)
  
  def critic_fn(inputs):
    model = IMPALANetwork_multihead_attention_item_aware(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        num_heads=num_heads,
        key_size=key_size)
    return model.critic(inputs)
  
  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )

def make_network_attention_multihead_self_supervision(environment_spec: EnvironmentSpec,
                                feature_extractor: hk.Module,
                                recurrent_dim: int = 128,
                                positional_embedding: Optional[str] = None,
                                add_selection_vec: bool = False,                            
                                num_heads: int = 4,
                                key_size: int = 64):   
  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention_self_supervision(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        num_heads=num_heads,
        key_size=key_size)
    return model(inputs, state) 

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_multihead_attention_self_supervision(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        num_heads=num_heads,
        key_size=key_size)
    return model.initial_state(batch_size)
  
  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_multihead_attention_self_supervision(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        num_heads=num_heads,
        key_size=key_size)
    return model.unroll(inputs, state)
  
  def critic_fn(inputs):
    model = IMPALANetwork_multihead_attention_self_supervision(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        num_heads=num_heads,
        key_size=key_size)
    return model.critic(inputs)
  
  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )
                                                      
def make_network_impala_cnn_visualization(environment_spec: EnvironmentSpec,
                   feature_extractor: hk.Module,
                   recurrent_dim: int = 128,
                   ):
  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetworkCNNVis(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetworkCNNVis(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetworkCNNVis(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.unroll(inputs, state)

  def critic_fn(inputs):
    model = IMPALANetworkCNNVis(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.critic(inputs)

  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )


class MultiHeadAttentionLayer(hk.Module):
  def __init__(
      self,
      num_heads: int,
      key_size_per_head: int,
      positional_embedding='learnable',
      add_selection_vec=False,
      attn_enhance_multiplier: float = 0.0,
      attn_enhance_head_indices: List[int] = [],
      dropout_rate: float = 0.0,
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

  def __call__(self, query, key, value, enhance_map=jnp.zeros((1, 121))):
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
      
    q_proj = hk.Linear(self.model_dim)(query)  # [B, model_dim]
    k_proj = hk.Linear(self.model_dim)(key)    # [B, N, model_dim]
    v_proj = hk.Linear(self.model_dim)(value)  # [B, N, model_dim]

    # if self.add_selection_vec:
    #   selection_vec = hk.get_parameter("selection_vector", shape=[self.model_dim], init=hk.initializers.TruncatedNormal(stddev=0.02))
    #   q_proj += selection_vec[None, :]

    # Split into heads
    grid_dim, embed_dim = k_proj.shape[-2], k_proj.shape[-1]
    q = q_proj.reshape(-1, self.num_heads, self.key_size_per_head)  # [B, H, K]
    q = q[:, :, None, :] # [B, H, 1, K]
    k = k_proj.reshape(-1, grid_dim, self.num_heads, self.key_size_per_head)  # [B, N, H, K]
    k = jnp.transpose(k, (0, 2, 1, 3))  # [B, H, N, K]
    v = v_proj.reshape(-1, grid_dim, self.num_heads, self.key_size_per_head)  # [B, N, H, K]
    v = jnp.transpose(v, (0, 2, 1, 3))  # [B, H, N, K]

    # Scaled dot-product attention
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).squeeze(2)  # [B, H, N]

    # if self.attn_enhance_multiplier != 0:
    #   enhance_map = enhance_map.reshape((-1, N))  # [B, N]
    #   max_scores = jnp.max(scores, axis=-1, keepdims=True)  # [B, H, 1]
    #   enhanced_scores = self.attn_enhance_multiplier * max_scores  # [B, H, 1]
    #   # Create a mask for enhanced heads: [H], 1 for enhance, 0 for others
    #   enhance_mask = jnp.isin(jnp.arange(self.num_heads), self.attn_enhance_head_indices).astype(jnp.float32)  # [H]
    #   enhance_mask = enhance_mask[None, :, None]  # [1, H, 1] for broadcasting
    #   # Broadcast enhance_map to match scores shape: [B, 1, N]
    #   enhance_map = enhance_map[:, None, :]
    #   # Apply enhancement only for selected heads and enhance_map positions
    #   scores = jnp.where(
    #       (enhance_map == 1) & (enhance_mask == 1),
    #       enhanced_scores,
    #       scores
    #   )

    scores = scores / jnp.sqrt(self.key_size_per_head)  # Scale scores
    weights = jax.nn.softmax(scores / self.temperature, axis=-1)  # [B, H, N]
    # if self.dropout_rate > 0.0 and self.is_training:
    #   weights = hk.dropout(hk.next_rng_key(), self.dropout_rate, weights)

    # Weighted sum
    output = jnp.einsum("bhn,bhnk->bhk", weights, v)  # [B, H, K]

    #output += jnp.mean(v,axis = 2) # residue connection 

    output = output.reshape(output.shape[0], -1)  # [B, model_dim]
    return output, weights

class MultiHeadAttentionDisturbLayer(hk.Module):
  def __init__(
      self,
      num_heads: int,
      key_size_per_head: int,
      positional_embedding=None,
      add_selection_vec=False,
      attn_enhance_multiplier: float = 0.0,
      disturb_head_indices: List[int] = [],
  ):
    super().__init__(name="multihead_attention_layer")
    self.num_heads = num_heads
    self.key_size_per_head = key_size_per_head  # per head
    self.positional_embedding = positional_embedding
    self.attn_enhance_multiplier = attn_enhance_multiplier
    self.add_selection_vec = add_selection_vec
    self.model_dim = num_heads * key_size_per_head
    if len(disturb_head_indices) > self.num_heads:
      raise ValueError("disturb_head_indices cannot be longer than num_heads")
    self.disturb_head_indices = np.array(disturb_head_indices, dtype=int)

  def __call__(self, query, key, value, enhance_map=jnp.zeros((1, 121))):
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

    # Linear projections
    q_proj = hk.Linear(self.model_dim)(query)  # [B, model_dim]
    k_proj = hk.Linear(self.model_dim)(key)    # [B, N, model_dim]
    v_proj = hk.Linear(self.model_dim)(value)  # [B, N, model_dim]

    if self.add_selection_vec:
      selection_vec = hk.get_parameter("selection_vector", shape=[self.model_dim], init=hk.initializers.TruncatedNormal(stddev=0.02))
      q_proj += selection_vec[None, :]

    # Split into heads
    grid_dim, embed_dim = k_proj.shape[-2], k_proj.shape[-1]
    q = q_proj.reshape(-1, self.num_heads, self.key_size_per_head)  # [B, H, K]
    q = q[:, :, None, :] # [B, H, 1, K]
    k = k_proj.reshape(-1, grid_dim, self.num_heads, self.key_size_per_head)  # [B, N, H, K]
    k = jnp.transpose(k, (0, 2, 1, 3))  # [B, H, N, K]
    v = v_proj.reshape(-1, grid_dim, self.num_heads, self.key_size_per_head)  # [B, N, H, K]
    v = jnp.transpose(v, (0, 2, 1, 3))  # [B, H, N, K]

    # Scaled dot-product attention
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).squeeze(2)  # [B, H, N]

    # === Disturb specified heads, vectorized ===    
    # Make a boolean mask [H] with True at disturbed heads
    mask = jnp.zeros((self.num_heads,), dtype=bool).at[self.disturb_head_indices].set(True)
    mask = mask[None, :, None]  # [1, H, 1] to broadcast
    all_ones = jnp.ones((B, self.num_heads, N))  # [B, H, N]
    scores = jnp.where(mask, all_ones, scores)
    
    # Replace scores of some head with other head's scores
    # third_head = scores[:, 2:3, :]
    # expanded = jnp.broadcast_to(third_head, scores.shape)  # shape: [B, H, N]
    # scores = jnp.where(mask, expanded, scores)
    
    # if self.attn_enhance_multiplier != 0:
    #   enhance_map = enhance_map.reshape((-1, N))  # [B, N]
    #   max_scores = jnp.max(scores, axis=-1, keepdims=True)  # [B, H, 1]
    #   enhanced_scores = self.attn_enhance_multiplier * max_scores
    #   scores = jnp.where(enhance_map[:, None, :] == 1, enhanced_scores, scores)

    scores = scores / jnp.sqrt(self.key_size_per_head * self.num_heads)  # Scale scores
    weights = jax.nn.softmax(scores, axis=-1)  # [B, H, N]

    # Weighted sum
    output = jnp.einsum("bhn,bhnk->bhk", weights, v)  # [B, H, K]
    output = output.reshape(output.shape[0], -1)  # [B, model_dim]
    return output, weights

class AttentionLayer(hk.Module):
  """Attention layer between CNN and LSTM."""
  def __init__(
      self, 
      key_size: int, 
      positional_embedding = None, 
      add_selection_vec = False,
      attn_enhance_multiplier: float = 0.0,
    ):
    """
    Args:
      key_size: Hidden dim of the key and value vectors
      positional_embedding: Type of positional embedding to use
      attn_enhancement_multiplier: Multiplier for enhancing the attention scores for specific locations
    """
    super().__init__(name="attention_layer")
    self.key_size = key_size
    self.positional_embedding = positional_embedding
    self.attn_enhance_multiplier = attn_enhance_multiplier
    self.add_selection_vec = add_selection_vec
  def __call__(self, query, key, value, enhance_map=None):
    # query: LSTM hidden state [B, H] H for hidden state dimension 
    # key, value: CNN features [B, 121, K] K for key/value dimension
    if jnp.ndim(query) == 1:
      # Add batch dimension to query
      query = jnp.expand_dims(query, axis=0)
    if jnp.ndim(key) == 2:
      # Add sequence dimension to key/value
      key = jnp.expand_dims(key, axis=0)
      value = jnp.expand_dims(value, axis=0)
    if self.positional_embedding == 'fixed':
      # # Add fixed CoordConv positional embedding [H, W, 2]
      B, N, _ = key.shape
      H = 11
      y = jnp.linspace(-1.0, 1.0, H)
      x = jnp.linspace(-1.0, 1.0, H) # assuming square grid, H = W
      yy, xx = jnp.meshgrid(y, x, indexing="ij")  # shape [H, W]
      coords = jnp.stack([yy, xx], axis=-1)       # [H, W, 2]
      coords = jnp.reshape(coords, [N, 2])         # [121, 2]
      coords = jnp.broadcast_to(coords[None, ...], [B, N, 2])  # [B, 121, 2]
      key = jnp.concatenate([key, coords], axis=-1)    # [B, 121, C+2]
      value = jnp.concatenate([value, coords], axis=-1)  # [B, 121, C+2] 
    elif self.positional_embedding == 'learnable':
      # learnable positional embedding [121, 64]
      #jax.debug.print('learnable positional embedding')
      pos_emb = hk.get_parameter("pos_embedding", shape=[key.shape[1], self.key_size], init=hk.initializers.TruncatedNormal(stddev=0.02))
      # Broadcast to batch size
      key += pos_emb[None, :, :]
      value += pos_emb[None, :, :]
    # else:
    #   jax.debug.print('no positional embedding')
    #   jax.debug.print(self.positional_embedding)

    ## Project query, key, value to same dimension
    query = hk.Linear(self.key_size)(query)  # [B, K]
    key = hk.Linear(self.key_size)(key)      # [B, 121, K]
    value = hk.Linear(self.key_size)(value)  # [B, 121, K]
    if self.add_selection_vec:
      # add a learnable selection vector to query 
      selection_vec = hk.get_parameter("selection_vector", shape=[self.key_size], init=hk.initializers.TruncatedNormal(stddev=0.02))
      # Broadcast to batch size
      query += selection_vec[None, :]  # [B, K]
    # Expand query dims for broadcasting
    query = jnp.expand_dims(query, axis=1)       # [B, 1, K]
    # Compute attention scores
    scores = jnp.einsum('bik,bjk->bij', query, key)  # [B, 1, 121]
    scores = jnp.squeeze(scores, axis=1)             # [B, 121]
    scores = scores / jnp.sqrt(self.key_size)
    weights = jax.nn.softmax(scores, axis=-1)        # [B, 121]
    
    # Apply attention
    output = jnp.einsum('bi,bik->bk', weights, value)  # [B, K]
    return output, weights

class AttentionItemAwareLayer(hk.Module):
  """Attention layer between CNN and LSTM."""
  def __init__(
      self, 
      key_size: int, 
      positional_embedding = None, 
      add_selection_vec = False,
      attn_enhance_multiplier: float = 0.0,
    ):
    """
    Args:
      key_size: Hidden dim of the key and value vectors
      positional_embedding: Type of positional embedding to use
      attn_enhancement_multiplier: Multiplier for enhancing the attention scores for specific locations
    """
    super().__init__(name="attention_layer")
    self.key_size = key_size
    self.positional_embedding = positional_embedding
    self.attn_enhance_multiplier = attn_enhance_multiplier
    self.add_selection_vec = add_selection_vec
  def __call__(self, query, key, value, enhance_map):
    # query: LSTM hidden state [B, H] H for hidden state dimension 
    # key, value: CNN features [B, 121, K] K for key/value dimension
    if jnp.ndim(query) == 1:
      # Add batch dimension to query
      query = jnp.expand_dims(query, axis=0)
    if jnp.ndim(key) == 2:
      # Add sequence dimension to key/value
      key = jnp.expand_dims(key, axis=0)
      value = jnp.expand_dims(value, axis=0)
    if self.positional_embedding == 'fixed':
      # # Add fixed CoordConv positional embedding [H, W, 2]
      B, N, _ = key.shape
      H = 11
      y = jnp.linspace(-1.0, 1.0, H)
      x = jnp.linspace(-1.0, 1.0, H) # assuming square grid, H = W
      yy, xx = jnp.meshgrid(y, x, indexing="ij")  # shape [H, W]
      coords = jnp.stack([yy, xx], axis=-1)       # [H, W, 2]
      coords = jnp.reshape(coords, [N, 2])         # [121, 2]
      coords = jnp.broadcast_to(coords[None, ...], [B, N, 2])  # [B, 121, 2]
      key = jnp.concatenate([key, coords], axis=-1)    # [B, 121, C+2]
      value = jnp.concatenate([value, coords], axis=-1)  # [B, 121, C+2] 
    elif self.positional_embedding == 'learnable':
      # learnable positional embedding [121, 64]
      #jax.debug.print('learnable positional embedding')
      pos_emb = hk.get_parameter("pos_embedding", shape=[key.shape[1], self.key_size], init=hk.initializers.TruncatedNormal(stddev=0.02))
      # Broadcast to batch size
      key += pos_emb[None, :, :]
      value += pos_emb[None, :, :]
    # else:
    #   jax.debug.print('no positional embedding')
    #   jax.debug.print(self.positional_embedding)

    ## Project query, key, value to same dimension
    query = hk.Linear(self.key_size)(query)  # [B, K]
    key = hk.Linear(self.key_size)(key)      # [B, 121, K]
    value = hk.Linear(self.key_size)(value)  # [B, 121, K]
    if self.add_selection_vec:
      # add a learnable selection vector to query 
      selection_vec = hk.get_parameter("selection_vector", shape=[self.key_size], init=hk.initializers.TruncatedNormal(stddev=0.02))
      # Broadcast to batch size
      query += selection_vec[None, :]  # [B, K]
    # Expand query dims for broadcasting
    query = jnp.expand_dims(query, axis=1)       # [B, 1, K]
    # Compute attention scores
    scores = jnp.einsum('bik,bjk->bij', query, key)  # [B, 1, 121]
    scores = jnp.squeeze(scores, axis=1)             # [B, 121]
   
    # Enhance attention scores for specific locations
    # enhance_map = enhance_map.reshape((-1, enhance_map.shape[-1]*enhance_map.shape[-2]))  # [B, 121]
    # max_scores = jnp.max(scores, axis=-1, keepdims=True)  # [B, 1]
    # enhancement = self.attn_enhance_multiplier * enhance_map * max_scores # [B, 121]
    # scores += enhancement  # [B, 121]
    # # Enhance attention scores for specific locations
    enhance_map = enhance_map.reshape((-1, enhance_map.shape[-1] * enhance_map.shape[-2]))  # [B, 121]
    max_scores = jnp.max(scores, axis=-1, keepdims=True)  # [B, 1]
    enhanced_scores = self.attn_enhance_multiplier * max_scores  # [B, 1]
    # Use `jnp.where` to set scores where enhance_map == 1
    scores = jnp.where(enhance_map == 1, enhanced_scores, scores)  # [B, 121]
    scores = scores / jnp.sqrt(self.key_size)
    weights = jax.nn.softmax(scores, axis=-1)        # [B, 121]
    
    # Apply attention
    output = jnp.einsum('bi,bik->bk', weights, value)  # [B, K]
    return output, weights

class AttentionLayerSpatial(hk.Module): 
  def __init__(self, key_size: int, add_selection_vec = False):
    super().__init__(name="attention_layer_spatial")
    self.key_size = key_size
    self.add_selection_vec = add_selection_vec
  def __call__(self, query, key, value):
    # query: LSTM hidden state [B, H] H for hidden state dimension 
    # key, value: CNN features [B, 121, K] K for key/value dimension
    if jnp.ndim(query) == 1:
      # Add batch dimension to query
      query = jnp.expand_dims(query, axis=0)
    if jnp.ndim(key) == 2:
      # Add sequence dimension to key/value
      key = jnp.expand_dims(key, axis=0)
      value = jnp.expand_dims(value, axis=0)
    
    ## Project query, key to same dimension
    query = hk.Linear(self.key_size)(query)  # [B, K]
    key = hk.Linear(self.key_size)(key)      # [B, 121, K]
    
    if self.add_selection_vec:
      # add a learnable selection vector to query 
      selection_vec = hk.get_parameter("selection_vector", shape=[self.key_size], init=hk.initializers.TruncatedNormal(stddev=0.02))
      # Broadcast to batch size
      query += selection_vec[None, :]  # [B, K]
    
    # Expand query dims for broadcasting
    query = jnp.expand_dims(query, axis=1)       # [B, 1, K]
    # Compute attention scores
    scores = jnp.einsum('bik,bjk->bij', query, key)  # [B, 1, 121]
    scores = jnp.squeeze(scores, axis=1)             # [B, 121]
    scores = scores / jnp.sqrt(self.key_size)
    weights = jax.nn.softmax(scores, axis=-1)        # [B, 121]
    weights = jnp.expand_dims(weights, axis=-1)      # [B, 121, 1]
    
    # Apply attention pixel-wise
    output = value * weights # [B, 121, K]
    B, H_square, K = output.shape
    H = 11 # TODO: make this dynamic
    output = jnp.reshape(output, (B, H, H, K)) # [B, 11, 11, K]
    return output, output

class AttentionLayerTanh(hk.Module): 
  """Attention layer between CNN and LSTM from https://github.com/AaronCCWong/Show-Attend-and-Tell/blob/master/attention.py"""
  def __init__(self, key_size: int, positional_embedding: bool = True):
    super().__init__(name="attention_tanh_layer")
    self.key_size = key_size
    self.positional_embedding = positional_embedding
    self.to_u = hk.Linear(key_size, name="to_u")
    self.to_w = hk.Linear(key_size, name="to_w")
    self.to_v = hk.Linear(1, name="to_v")
    self.tanh = jax.nn.tanh
    self.softmax = jax.nn.softmax

  def __call__(self, query, key, value):
    # query: LSTM hidden state [B, H] H for hidden state dimension 
    # key, value: CNN features [B, 121, K] K for key/value dimension
    if jnp.ndim(query) == 1:
      # Add batch dimension to query
      query = jnp.expand_dims(query, axis=0)
    if jnp.ndim(key) == 2:
      # Add sequence dimension to key/value
      key = jnp.expand_dims(key, axis=0)
      value = jnp.expand_dims(value, axis=0)
    if self.positional_embedding:
      # Add fixed CoordConv positional embedding [H, W, 2]
      B, N, _ = key.shape
      H = 11
      y = jnp.linspace(-1.0, 1.0, H)
      x = jnp.linspace(-1.0, 1.0, H) # assuming square grid, H = W
      yy, xx = jnp.meshgrid(y, x, indexing="ij")  # shape [H, W]
      coords = jnp.stack([yy, xx], axis=-1)       # [H, W, 2]
      coords = jnp.reshape(coords, [N, 2])         # [121, 2]
      coords = jnp.broadcast_to(coords[None, ...], [B, N, 2])  # [B, 121, 2]
      key = jnp.concatenate([key, coords], axis=-1)    # [B, 121, C+2]
      value = jnp.concatenate([value, coords], axis=-1)  # [B, 121, C+2] 

    u_hidden = self.to_u(query)          # [B, K]
    u_hidden = jnp.expand_dims(u_hidden, axis=1)       # [B, 1, K]
    w_key = self.to_w(key)                # [B, 121, K]
    attn = self.tanh(w_key + u_hidden)  # [B, 121, K]
    scores = self.to_v(attn)             # [B, 121, 1]
    scores = jnp.squeeze(scores, axis=-1)  # [B, 121]
    weights = self.softmax(scores, axis=-1)        # [B, 121]
    output = jnp.einsum('bi,bik->bk', weights, value)  # [B, K]
    return output, weights
  
class PostAttnCNN(hk.Module):
  def __init__(self):
    super().__init__(name="PostAttnCNN")
    self._cnn = hk.Sequential([
        hk.Conv2D(32, [4, 4], 1, padding="VALID"),
        jax.nn.relu,
    ])
    self._ff = hk.Sequential([
        hk.Linear(64),
        jax.nn.relu,
        hk.Linear(64),
        jax.nn.relu,
    ])

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError("Expected input BHWC or HWC. Got rank %d" % inputs_rank)

    outputs = self._cnn(inputs)

    if batched_inputs:
      outputs = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      outputs = jnp.reshape(outputs, [-1])  # [D]

    outputs = self._ff(outputs)
    return outputs
  
def zero_state_lstm(lstm: hk.LSTM):
    def wrapped(inputs, state):
        if inputs.ndim == 1:
          batch_size = None
        else:
          batch_size = state.hidden.shape[0]
        zero_state = lstm.initial_state(batch_size)
        return lstm(inputs, zero_state)
    return wrapped
    
class IMPALANetwork_attention(hk.RNNCore):
  """Network architecture as described in MeltingPot paper"""

  def __init__(
      self, 
      num_actions,
      recurrent_dim, 
      feature_extractor, 
      positional_embedding=None, 
      add_selection_vec=False, 
      attn_enhance_multiplier: float = 0.0,
      use_layer_norm: bool = False,
      zero_state: bool = False,):
    super().__init__(name="impala_network")
    self.num_actions = num_actions
    self.use_layer_norm = use_layer_norm
    self._embed = feature_extractor(num_actions)
    self._attention = AttentionLayer(
      key_size=64, 
      positional_embedding=positional_embedding, 
      add_selection_vec=add_selection_vec, 
      attn_enhance_multiplier=attn_enhance_multiplier,)
    if self.use_layer_norm:
      self._attn_layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="attn_layer_norm")
    self._lstm_core = hk.LSTM(recurrent_dim)
    if zero_state:
      self._recurrent = zero_state_lstm(self._lstm_core)
    else:
      self._recurrent = self._lstm_core
    self._policy_layer = hk.Linear(num_actions, name="policy")
    self._value_layer = hk.Linear(1, name="value_layer")
    
  def __call__(self, inputs, state: hk.LSTMState):
    embedding = self._embed(inputs) # [B, 121, F]
    # Apply attention between CNN features and LSTM hidden state
    attended, attn_weights = self._attention(state.hidden, embedding, embedding)
    if self.use_layer_norm:
      attended = self._attn_layer_norm(attended)

    # extract other observations
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
    op, new_state = self._recurrent(combined, state)
    new_state = ContextState(cell=new_state.cell, hidden=new_state.hidden, buffer=jnp.array([1], dtype=jnp.int32))
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, attn_weights), new_state

  # def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
  #   return self._lstm_core.initial_state(batch_size)
  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    lstm_state = self._lstm_core.initial_state(batch_size)
    dummy_buffer = jnp.array([1], dtype=jnp.int32)
    return ContextState(cell=lstm_state.cell, hidden=lstm_state.hidden, buffer=dummy_buffer)

  def unroll(self, inputs, state: hk.LSTMState):
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    op = self._embed(inputs)
    flatten_op = jnp.reshape(op, (op.shape[0], -1))  # [B, ...]
    self.op_shape = op.shape  # save the original shape for later use
    #jax.debug.print("op shape: {}", op.shape)
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

    def _step(input_t, state):
      # Slice out the flattened input for the current time step
      attn_input = input_t[:self.flatten_op_dim]
      # Reshape it back to the original shape
      attn_input = jnp.reshape(attn_input, self.op_shape[1:]) # skip batch/time dimension
      # Apply attention
      attended, attn_weights = self._attention(state.hidden, attn_input, attn_input)
      if self.use_layer_norm:
        attended = self._attn_layer_norm(attended)
      # Combine with the rest of the input
      rest_input = input_t[self.flatten_op_dim:]
      if rest_input.ndim < attended.ndim:
        attended = jnp.squeeze(attended, axis=0)
      combined_input = jnp.concatenate([attended, rest_input], axis=-1)
      # Go through the recurrent layer    
      output, new_state = self._recurrent(combined_input, state)
      new_state = ContextState(cell=new_state.cell, hidden=new_state.hidden, buffer=jnp.array([1], dtype=jnp.int32))
      return (output,attn_weights), new_state

    # Unroll through time
    optuple, new_states = hk.static_unroll(
        _step,
        combined, 
        state
    )
    op,attn_weights = optuple
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, attn_weights), new_states

  def critic(self, inputs):
    return jnp.squeeze(self._value_layer(inputs), axis=-1)

class IMPALANetwork_attention_spatial(IMPALANetwork_attention):
  def __init__(self, num_actions, recurrent_dim, feature_extractor, add_selection_vec=False):
    super().__init__(num_actions, recurrent_dim, feature_extractor, positional_embedding=None)
    self.num_actions = num_actions
    self._embed = feature_extractor(num_actions)
    self._attention = AttentionLayerSpatial(key_size=16, add_selection_vec=add_selection_vec)
    self._post_attn_cnn = PostAttnCNN()
    self._recurrent = hk.LSTM(recurrent_dim)
    self._policy_layer = hk.Linear(num_actions, name="policy")
    self._value_layer = hk.Linear(1, name="value_layer")
    
  def __call__(self, inputs, state: hk.LSTMState):
    embedding = self._embed(inputs) # [B, 11, 11, F]
    # Apply attention between CNN features and LSTM hidden state
    attended, attn_weights = self._attention(state.hidden, embedding, embedding) # [B, 11, 11, F]
    attended = self._post_attn_cnn(attended)  # [B, 64]

    # extract other observations
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
    op, new_state = self._recurrent(combined, state)
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, attn_weights), new_state
  
  def unroll(self, inputs, state: hk.LSTMState):
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    op = self._embed(inputs)
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

    def _step(input_t, state):
      # Slice out the flattened input for the current time step
      attn_input = input_t[:self.flatten_op_dim]
      # Reshape it back to the original shape
      attn_input = jnp.reshape(attn_input, self.op_shape[1:]) # skip batch/time dimension
      # Apply attention
      attended, attn_weights = self._attention(state.hidden, attn_input, attn_input)
      # Apply post-attention CNN
      attended = self._post_attn_cnn(attended)
      # Combine with the rest of the input
      rest_input = input_t[self.flatten_op_dim:]
      if rest_input.ndim < attended.ndim:
        attended = jnp.squeeze(attended, axis=0)
      combined_input = jnp.concatenate([attended, rest_input], axis=-1)
      # Go through the recurrent layer
      output, new_state = self._recurrent(combined_input, state)
      return output, new_state

    # Unroll through time
    op, new_states = hk.static_unroll(
        _step,
        combined, 
        state
    )
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, None), new_states

class IMPALANetwork_attention_tanh(IMPALANetwork_attention):
  def __init__(self, num_actions, recurrent_dim, feature_extractor, positional_embedding=None):
    super().__init__(num_actions, recurrent_dim, feature_extractor, positional_embedding)
    self.num_actions = num_actions
    self._embed = feature_extractor(num_actions)
    self._attention = AttentionLayerTanh(key_size=64, positional_embedding=positional_embedding)
    self._recurrent = hk.LSTM(recurrent_dim)
    self._policy_layer = hk.Linear(num_actions, name="policy")
    self._value_layer = hk.Linear(1, name="value_layer")

class IMPALANetwork_attention_item_aware(IMPALANetwork_attention):
  def __init__(
      self, 
      num_actions, 
      recurrent_dim, 
      feature_extractor, 
      positional_embedding=None,
      attn_enhance_multiplier=1.0,
      add_selection_vec=False, 
      use_layer_norm: bool = False
    ):
    super().__init__(
      num_actions=num_actions,
      recurrent_dim=recurrent_dim,
      feature_extractor=feature_extractor,
      positional_embedding=positional_embedding,
      add_selection_vec=add_selection_vec,
      attn_enhance_multiplier=attn_enhance_multiplier,
      use_layer_norm=use_layer_norm
    )
    self._attention = AttentionItemAwareLayer(
      key_size=64,
      positional_embedding=positional_embedding, 
      add_selection_vec=add_selection_vec,
      attn_enhance_multiplier=attn_enhance_multiplier,
    )

  def __call__(self, inputs, state: hk.LSTMState):
    embedding = self._embed(inputs) # [B, 121, F]
    obs = inputs["observation"]

    enhance_mask = obs["OBJECTS_IN_VIEW"]  # [B, 11, 11]
    attended, attn_weights = self._attention(state.hidden, embedding, embedding, enhance_mask)

    # extract other observations
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
    op, new_state = self._recurrent(combined, state)
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, attn_weights), new_state
  
  def unroll(self, inputs, state: hk.LSTMState):
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    op = self._embed(inputs)
    flatten_op = jnp.reshape(op, (op.shape[0], -1))  # [B, ...]
    self.op_shape = op.shape  # save the original shape for later use
    self.flatten_op_dim = flatten_op.shape[-1]  # save the flattened dimension for later use
    # extract other observations
    obs = inputs["observation"]
    inventory, ready_to_shoot, enhance_mask = obs["INVENTORY"], obs["READY_TO_SHOOT"], obs["OBJECTS_IN_VIEW"]
    flatten_enhance_mask = jnp.reshape(enhance_mask, (-1, enhance_mask.shape[-1]*enhance_mask.shape[-2]))  # [B, 121]
    self.enhance_mask_shape = enhance_mask.shape  # save the original shape for later use
    self.flatten_enhance_mask_dim = flatten_enhance_mask.shape[-1]  # save the flattened dimension for later use
    # Do a one-hot embedding of the actions.
    action = jax.nn.one_hot(
        inputs["action"], num_classes=self.num_actions)  # [B, A]
    # Add dummy trailing dimensions to rewards if necessary.
    while ready_to_shoot.ndim < inventory.ndim:
      ready_to_shoot = jnp.expand_dims(ready_to_shoot, axis=-1)
    # fix for dynamic_unroll with reverb sampled data
    # state = hk.LSTMState(state.hidden, state.cell)
    combined = jnp.concatenate(
        [flatten_op, flatten_enhance_mask, ready_to_shoot, inventory, action], 
        axis=-1
    ) 

    def _step(input_t, state):
      # Slice out the flattened input for the current time step
      attn_input = input_t[:self.flatten_op_dim]
      enhance_mask_input = input_t[self.flatten_op_dim:self.flatten_op_dim + self.flatten_enhance_mask_dim]
      # Reshape it back to the original shape
      attn_input = jnp.reshape(attn_input, self.op_shape[1:]) # skip batch/time dimension
      enhance_mask_input = jnp.reshape(enhance_mask_input, (self.enhance_mask_shape[-2], self.enhance_mask_shape[-1]))
      # Apply attention
      attended, attn_weights = self._attention(state.hidden, attn_input, attn_input, enhance_mask_input)
      # Combine with the rest of the input
      rest_input = input_t[self.flatten_op_dim + self.flatten_enhance_mask_dim:]
      if rest_input.ndim < attended.ndim:
        attended = jnp.squeeze(attended, axis=0)
      combined_input = jnp.concatenate([attended, rest_input], axis=-1)
      # Go through the recurrent layer
      output, new_state = self._recurrent(combined_input, state)
      return output, new_state

    # Unroll through time
    op, new_states = hk.static_unroll(
        _step,
        combined, 
        state
    )
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, None), new_states

class IMPALANetwork_multihead_attention(IMPALANetwork_attention):
  # TODO: Since call and unroll are not overridden, this class is
  # not supporting item-aware attention for now.
  def __init__(
      self, 
      num_actions,
      recurrent_dim, 
      feature_extractor, 
      num_heads=4,
      key_size=64,
      positional_embedding=None, 
      add_selection_vec=False, 
      attn_enhance_multiplier=0.0,
      use_layer_norm=False,):
    super().__init__(
      num_actions, 
      recurrent_dim, 
      feature_extractor, 
      positional_embedding, 
      add_selection_vec, 
      attn_enhance_multiplier,
      use_layer_norm)
    self._attention = MultiHeadAttentionLayer(
        num_heads=num_heads,
        key_size_per_head=key_size // num_heads,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier)

class IMPALANetwork_multihead_attention_disturb(IMPALANetwork_attention):
  # TODO: Since call and unroll are not overridden, this class is
  # not supporting item-aware attention for now.
  def __init__(
      self, 
      num_actions,
      recurrent_dim, 
      feature_extractor, 
      num_heads=4,
      key_size=64,
      positional_embedding=None, 
      add_selection_vec=False, 
      attn_enhance_multiplier=0.0,
      use_layer_norm=False,
      disturb_head_indices=[]):
    super().__init__(
      num_actions, 
      recurrent_dim, 
      feature_extractor, 
      positional_embedding, 
      add_selection_vec, 
      attn_enhance_multiplier,
      use_layer_norm)
    self._attention = MultiHeadAttentionDisturbLayer(
        num_heads=num_heads,
        key_size_per_head=key_size // num_heads,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        disturb_head_indices=disturb_head_indices)

class IMPALANetwork_multihead_attention_item_aware(IMPALANetwork_attention):
  def __init__(
      self, 
      num_actions,
      recurrent_dim, 
      feature_extractor, 
      num_heads=4,
      key_size=64,
      positional_embedding=None, 
      add_selection_vec=False, 
      use_layer_norm=False,):
    super().__init__(
      num_actions,
      recurrent_dim,
      feature_extractor,
      positional_embedding=positional_embedding,
      add_selection_vec=add_selection_vec,
      use_layer_norm=use_layer_norm)
    self._attention = MultiHeadAttentionLayer(
        num_heads=num_heads,
        key_size_per_head=key_size // num_heads,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=0)
    
  def __call__(self, inputs, state: hk.LSTMState):
    # TODO: current version does not support test time attention enhancement
    embedding = self._embed(inputs) # [B, 121, F]
    obs = inputs["observation"]

    objects_in_view = obs["OBJECTS_IN_VIEW"]  # [B, n_item_types, 11, 11]
    attended, attn_weights = self._attention(state.hidden, embedding, embedding)

    # extract other observations
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
    op, new_state = self._recurrent(combined, state)
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, attn_weights), new_state
  
  def unroll(self, inputs, state: hk.LSTMState):
    """
      Efficient unroll that applies embeddings, MLP, & convnet in one pass.
      
      Same as normal IMPALANetwork_attention, but also return objects_in_view heatmaps from 
      observation, which is used for item-aware cross-entropy loss.
    """
    op = self._embed(inputs)
    flatten_op = jnp.reshape(op, (op.shape[0], -1))  # [B, ...]
    self.op_shape = op.shape  # save the original shape for later use
    self.flatten_op_dim = flatten_op.shape[-1]  # save the flattened dimension for later use
    # extract other observations
    obs = inputs["observation"]
    objects_in_view = obs["OBJECTS_IN_VIEW"]  # [B, n_item_types, 11, 11]
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

    def _step(input_t, state):
      # Slice out the flattened input for the current time step
      attn_input = input_t[:self.flatten_op_dim]
      # Reshape it back to the original shape
      attn_input = jnp.reshape(attn_input, self.op_shape[1:]) # skip batch/time dimension
      # Apply attention
      attended, attn_weights = self._attention(state.hidden, attn_input, attn_input)
      if self.use_layer_norm:
        attended = self._attn_layer_norm(attended)
      # Combine with the rest of the input
      rest_input = input_t[self.flatten_op_dim:]
      if rest_input.ndim < attended.ndim:
        attended = jnp.squeeze(attended, axis=0)
      combined_input = jnp.concatenate([attended, rest_input], axis=-1)
      # Go through the recurrent layer
      output, new_state = self._recurrent(combined_input, state)
      return (output, attn_weights), new_state

    # Unroll through time
    optuple, new_states = hk.static_unroll(
        _step,
        combined, 
        state
    )
    op, attn_weights = optuple
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, (attn_weights, objects_in_view)), new_states

class IMPALANetwork_multihead_attention_self_supervision(IMPALANetwork_attention):
  def __init__(
      self, 
      num_actions,
      recurrent_dim, 
      feature_extractor, 
      num_heads=4,
      key_size=64,
      positional_embedding=None, 
      add_selection_vec=False, 
      use_layer_norm=False,):
    super().__init__(
      num_actions,
      recurrent_dim,
      feature_extractor,
      positional_embedding=positional_embedding,
      add_selection_vec=add_selection_vec,
      use_layer_norm=use_layer_norm)
    self._attention = MultiHeadAttentionLayer(
        num_heads=num_heads,
        key_size_per_head=key_size // num_heads,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=0)
    
  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    lstm_state = self._lstm_core.initial_state(batch_size)
    dummy_buffer = jnp.array([1], dtype=jnp.int32)
    return ContextState(cell=lstm_state.cell, hidden=lstm_state.hidden, buffer=dummy_buffer)
    
  def __call__(self, inputs, state: hk.LSTMState):
  
    embedding, self_guidance_attn_map = self._embed(inputs) # [B, 121, F]
    obs = inputs["observation"]
    attended, attn_weights = self._attention(state.hidden, embedding, embedding)
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
    op, new_state = self._recurrent(combined, state)
    new_state = ContextState(cell=new_state.cell, hidden=new_state.hidden, buffer=jnp.array([1], dtype=jnp.int32))
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    #return (logits, value, op, attn_weights), new_state
    return (logits, value, op, self_guidance_attn_map), new_state
  
  
  def unroll(self, inputs, state: hk.LSTMState):
    """
      Efficient unroll that applies embeddings, MLP, & convnet in one pass.
      
      Same as normal IMPALANetwork_attention, but also return objects_in_view heatmaps from 
      observation, which is used for item-aware cross-entropy loss.
    """
    op, self_guidance_attn_map = self._embed(inputs) # [B, 121, F], [B, n_item_types, 11, 11]
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

    def _step(input_t, state):
      # Slice out the flattened input for the current time step
      attn_input = input_t[:self.flatten_op_dim]
      # Reshape it back to the original shape
      attn_input = jnp.reshape(attn_input, self.op_shape[1:]) # skip batch/time dimension
      # Apply attention
      attended, attn_weights = self._attention(state.hidden, attn_input, attn_input)
      if self.use_layer_norm:
        attended = self._attn_layer_norm(attended)
      # Combine with the rest of the input
      rest_input = input_t[self.flatten_op_dim:]
      if rest_input.ndim < attended.ndim:
        attended = jnp.squeeze(attended, axis=0)
      combined_input = jnp.concatenate([attended, rest_input], axis=-1)
      # Go through the recurrent layer
      output, new_state = self._recurrent(combined_input, state)
      new_state = ContextState(cell=new_state.cell, hidden=new_state.hidden, buffer=jnp.array([1], dtype=jnp.int32))
      return (output, attn_weights), new_state

    # Unroll through time
    optuple, new_states = hk.static_unroll(
        _step,
        combined, 
        state
    )
    op, attn_weights = optuple
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, (attn_weights, self_guidance_attn_map)), new_states

class IMPALANetwork_multihead_attention_enhance(IMPALANetwork_attention):
  def __init__(
      self, 
      num_actions,
      recurrent_dim, 
      feature_extractor, 
      num_heads=4,
      key_size=64,
      positional_embedding=None, 
      add_selection_vec=False, 
      use_layer_norm=False,
      attn_enhance_multiplier=1.0,
      attn_enhance_head_indices=[],
      attn_enhance_item_idx=0,):
    super().__init__(
      num_actions,
      recurrent_dim,
      feature_extractor,
      positional_embedding=positional_embedding,
      add_selection_vec=add_selection_vec,
      use_layer_norm=use_layer_norm)
    self._attention = MultiHeadAttentionLayer(
        num_heads=num_heads,
        key_size_per_head=key_size // num_heads,
        positional_embedding=positional_embedding,
        add_selection_vec=add_selection_vec,
        attn_enhance_multiplier=attn_enhance_multiplier,
        attn_enhance_head_indices=attn_enhance_head_indices)
    self.attn_enhance_item_idx = attn_enhance_item_idx  # index of the item to enhance attention on

  def __call__(self, inputs, state: hk.LSTMState):
    # TODO: current version does not support test time attention enhancement
    embedding = self._embed(inputs) # [B, 121, F]
    obs = inputs["observation"]

    objects_in_view = obs["OBJECTS_IN_VIEW"]  # [n_item_types, 11, 11]
    focal_objects = objects_in_view[self.attn_enhance_item_idx]  # [11, 11]
    focal_objects = jnp.reshape(focal_objects, (-1, focal_objects.shape[-1]*focal_objects.shape[-2]))  # [B, 121]
    attended, attn_weights = self._attention(state.hidden, embedding, embedding, enhance_map=focal_objects)

    # extract other observations
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
    op, new_state = self._recurrent(combined, state)
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, attn_weights), new_state
  
class IMPALANetwork(hk.RNNCore):
  """Network architecture as described in MeltingPot paper"""

  def __init__(self, num_actions, recurrent_dim, feature_extractor):
    super().__init__(name="impala_network")
    self.num_actions = num_actions
    self._embed = feature_extractor(num_actions)
    self._recurrent = hk.LSTM(recurrent_dim)
    self._policy_layer = hk.Linear(num_actions, name="policy")
    self._value_layer = hk.Linear(1, name="value_layer")

  def __call__(self, inputs, state: hk.LSTMState):
    emb = self._embed(inputs)
    op, new_state = self._recurrent(emb, state)
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, emb), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._recurrent.initial_state(batch_size)

  def unroll(self, inputs, state: hk.LSTMState):
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    emb = self._embed(inputs)

    # fix for dynamic_unroll with reverb sampled data
    # state = hk.LSTMState(state.hidden, state.cell)

    # unrolling the time dimension
    op, new_states = hk.static_unroll(self._recurrent, emb,
                                      state)  # , return_all_states=True)

    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, emb), new_states

  def critic(self, inputs):
    return jnp.squeeze(self._value_layer(inputs), axis=-1)

class IMPALANetworkCNNVis(IMPALANetwork):
  """Network architecture as described in MeltingPot paper"""

  def __init__(self, num_actions, recurrent_dim, feature_extractor):
    super().__init__(num_actions, recurrent_dim, feature_extractor)

  def __call__(self, inputs, state: hk.LSTMState):
    emb, cnn_attn = self._embed(inputs)
    op, new_state = self._recurrent(emb, state)
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, cnn_attn), new_state

  def unroll(self, inputs, state: hk.LSTMState):
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    emb, cnn_attn = self._embed(inputs)
    obs = inputs["observation"]
    objects_in_view = obs["OBJECTS_IN_VIEW"]  # [B, n_item_types, 11, 11]

    # fix for dynamic_unroll with reverb sampled data
    # state = hk.LSTMState(state.hidden, state.cell)

    # unrolling the time dimension
    op, new_states = hk.static_unroll(self._recurrent, emb,
                                      state)  # , return_all_states=True)

    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op, (cnn_attn, objects_in_view)), new_states
