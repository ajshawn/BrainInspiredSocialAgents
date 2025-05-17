from acme.jax import utils
from acme.specs import EnvironmentSpec
import haiku as hk
import jax.numpy as jnp
import jax
from marl.agents.networks import make_haiku_networks
from marl.agents.networks import make_haiku_networks_2

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

  def critic_fn(inputs):
    model = IMPALANetwork(
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

class AttentionLayer(hk.Module):
  """Attention layer between CNN and LSTM."""
  def __init__(self, key_size: int):
    super().__init__(name="attention_layer")
    self.key_size = key_size
      
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
    
    ## Project query, key, value to same dimension
    query = hk.Linear(self.key_size)(query)          # [B, K]
    key = hk.Linear(self.key_size)(key)              # [B, 121, K]
    value = hk.Linear(self.key_size)(value)          # [B, 121, K]
    
    # Expand query dims for broadcasting
    query = jnp.expand_dims(query, axis=1)       # [B, 1, K]
    # Compute attention scores
    scores = jnp.einsum('bik,bjk->bij', query, key)  # [B, 1, 121]
    scores = jnp.squeeze(scores, axis=1)             # [B, 121]
    scores = scores / jnp.sqrt(self.key_size)
    weights = jax.nn.softmax(scores, axis=-1)        # [B, 121]
    
    # Apply attention
    output = jnp.einsum('bi,bik->bk', weights, value)  # [B, K]
    return output


class IMPALANetwork(hk.RNNCore):
  """Network architecture as described in MeltingPot paper"""

  def __init__(self, num_actions, recurrent_dim, feature_extractor):
    super().__init__(name="impala_network")
    self.num_actions = num_actions
    self._embed = feature_extractor(num_actions)
    self._attention = AttentionLayer(key_size=64)
    self._recurrent = hk.LSTM(recurrent_dim)
    self._policy_layer = hk.Linear(num_actions, name="policy")
    self._value_layer = hk.Linear(1, name="value_layer")
  def __call__(self, inputs, state: hk.LSTMState):
    embedding = self._embed(inputs) # [B, 121, F]
    # Apply attention between CNN features and LSTM hidden state
    attended = self._attention(state.hidden, embedding, embedding)

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
    return (logits, value, embedding), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._recurrent.initial_state(batch_size)

  def unroll(self, inputs, state: hk.LSTMState):
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    op = self._embed(inputs)
    attended = self._attention(state.hidden, op, op)
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
    # fix for dynamic_unroll with reverb sampled data
    # state = hk.LSTMState(state.hidden, state.cell)
    combined = jnp.concatenate(
        [attended, ready_to_shoot, inventory, action], 
        axis=-1
    )
    
    # Unroll through time
    op, new_states = hk.static_unroll(
        self._recurrent,
        combined, 
        state
    )

    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op), new_states

  def critic(self, inputs):
    return jnp.squeeze(self._value_layer(inputs), axis=-1)
