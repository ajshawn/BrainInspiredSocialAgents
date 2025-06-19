from acme.jax import utils
from acme.specs import EnvironmentSpec
import haiku as hk
import jax.numpy as jnp
import jax
from marl.agents.networks import make_haiku_networks
from marl.agents.networks import make_haiku_networks_2

from typing import Optional

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
                   positional_embedding: Optional[str] = None):

  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding)
    return model(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork_attention(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor,
        positional_embedding=positional_embedding)
    return model.unroll(inputs, state)

  def critic_fn(inputs):
    model = IMPALANetwork_attention(
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

class AttentionLayer(hk.Module):
  """Attention layer between CNN and LSTM."""
  def __init__(self, key_size: int, positional_embedding = None):
    super().__init__(name="attention_layer")
    self.key_size = key_size
    self.positional_embedding = positional_embedding
    
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

class IMPALANetwork_attention(hk.RNNCore):
  """Network architecture as described in MeltingPot paper"""

  def __init__(self, num_actions, recurrent_dim, feature_extractor, positional_embedding=None):
    super().__init__(name="impala_network")
    self.num_actions = num_actions
    self._embed = feature_extractor(num_actions)
    self._attention = AttentionLayer(key_size=64, positional_embedding=positional_embedding)
    self._recurrent = hk.LSTM(recurrent_dim)
    self._policy_layer = hk.Linear(num_actions, name="policy")
    self._value_layer = hk.Linear(1, name="value_layer")
    
  def __call__(self, inputs, state: hk.LSTMState):
    embedding = self._embed(inputs) # [B, 121, F]
    # Apply attention between CNN features and LSTM hidden state
    attended, attn_weights = self._attention(state.hidden, embedding, embedding)

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
    return (logits, value, attn_weights), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._recurrent.initial_state(batch_size)

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
    return (logits, value, None), new_states

  def critic(self, inputs):
    return jnp.squeeze(self._value_layer(inputs), axis=-1)

class IMPALANetwork_attention_spatial(IMPALANetwork_attention):
  def __init__(self, num_actions, recurrent_dim, feature_extractor, add_selection_vec=False):
    super().__init__(num_actions, recurrent_dim, feature_extractor, positional_embedding=None)
    self.num_actions = num_actions
    self._embed = feature_extractor(num_actions)
    self._attention = AttentionLayerSpatial(key_size=64, add_selection_vec=add_selection_vec)
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
    return (logits, value, attn_weights), new_state
  
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
    return (logits, value, None), new_states

class IMPALANetwork_attention_tanh(IMPALANetwork_attention):
  def __init__(self, num_actions, recurrent_dim, feature_extractor, positional_embedding=True):
    super().__init__(num_actions, recurrent_dim, feature_extractor, positional_embedding)
    self.num_actions = num_actions
    self._embed = feature_extractor(num_actions)
    self._attention = AttentionLayerTanh(key_size=64, positional_embedding=positional_embedding)
    self._recurrent = hk.LSTM(recurrent_dim)
    self._policy_layer = hk.Linear(num_actions, name="policy")
    self._value_layer = hk.Linear(1, name="value_layer")

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
    return (logits, value, emb), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._recurrent.initial_state(batch_size)

  def unroll(self, inputs, state: hk.LSTMState):
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    emb = self._embed(inputs)

    # fix for dynamic_unroll with reverb sampled data
    # state = hk.LSTMState(state.hidden, state.cell)

    # unrolling the time dimension
    op, new_states = hk.static_unroll(self._recurrent, op,
                                      state)  # , return_all_states=True)

    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, emb), new_states

  def critic(self, inputs):
    return jnp.squeeze(self._value_layer(inputs), axis=-1)
