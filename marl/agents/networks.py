from typing import Any

from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.specs import EnvironmentSpec
import haiku as hk
import jax
import jax.numpy as jnp

from marl import types

# Useful type aliases
Images = jnp.ndarray


def make_haiku_networks(
    env_spec: EnvironmentSpec,
    forward_fn: Any,
    initial_state_fn: Any,
    unroll_fn: Any,
) -> types.RecurrentNetworks[types.RecurrentState]:
  """Builds functional network from recurrent model definitions."""
  # Make networks purely functional.
  forward_hk = hk.without_apply_rng(hk.transform(forward_fn))
  initial_state_hk = hk.without_apply_rng(hk.transform(initial_state_fn))
  unroll_hk = hk.without_apply_rng(hk.transform(unroll_fn))

  # Note: batch axis is not needed for the actors.
  dummy_obs = utils.zeros_like(env_spec.observations)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs)

  def unroll_init_fn(rng: networks_lib.PRNGKey,
                     initial_state: types.RecurrentState) -> hk.Params:
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state)

  return types.RecurrentNetworks(
      forward_fn=forward_hk.apply,
      unroll_fn=unroll_hk.apply,
      unroll_init_fn=unroll_init_fn,
      initial_state_fn=(lambda rng, batch_size=None: initial_state_hk.apply(
          initial_state_hk.init(rng), batch_size)),
  )


def make_haiku_networks_2(
    env_spec: EnvironmentSpec,
    forward_fn: Any,
    initial_state_fn: Any,
    unroll_fn: Any,
    critic_fn: Any,
) -> types.ActorCriticRecurrentNetworks[types.RecurrentState]:
  """Builds functional network from recurrent model definitions."""
  # Make networks purely functional.
  forward_hk = hk.without_apply_rng(hk.transform(forward_fn))
  initial_state_hk = hk.without_apply_rng(hk.transform(initial_state_fn))
  unroll_hk = hk.without_apply_rng(hk.transform(unroll_fn))
  critic_hk = hk.without_apply_rng(hk.transform(critic_fn))

  # Note: batch axis is not needed for the actors.
  dummy_obs = utils.zeros_like(env_spec.observations)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs)

  def unroll_init_fn(rng: networks_lib.PRNGKey,
                     initial_state: types.RecurrentState) -> hk.Params:
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state)

  return types.ActorCriticRecurrentNetworks(
      forward_fn=forward_hk.apply,
      critic_fn=critic_hk.apply,
      unroll_fn=unroll_hk.apply,
      unroll_init_fn=unroll_init_fn,
      initial_state_fn=(lambda rng, batch_size=None: initial_state_hk.apply(
          initial_state_hk.init(rng), batch_size)),
  )


class ArrayFE(hk.Module):

  def __init__(self, num_actions, hidden_dim=64):
    super().__init__("array_features")
    self.num_actions = num_actions
    self._layer = hk.Sequential([
        hk.Linear(hidden_dim),
        jax.nn.relu,
        hk.Linear(hidden_dim),
        jax.nn.relu,
    ])

  def __call__(self, inputs):
    op = self._layer(inputs["observation"]["agent_obs"])
    action = jax.nn.one_hot(inputs["action"], num_classes=self.num_actions)
    combined_op = jnp.concatenate([op, action], axis=-1)
    return combined_op


class ImageFE(hk.Module):

  def __init__(self, num_actions):
    super().__init__("image_features")
    self.num_actions = num_actions
    self._cnn = hk.Sequential([
        hk.Conv2D(16, [4, 4], 1, padding="VALID"),
        jax.nn.relu,
        hk.Conv2D(32, [4, 4], 1, padding="VALID"),
        jax.nn.relu,
    ])
    self._ff = hk.Sequential([
        hk.Linear(64),
        jax.nn.relu,
        hk.Linear(64),
        jax.nn.relu,
    ])

  def __call__(self, inputs) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs["observation"]["agent_obs"])
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError("Expected input BHWC or HWC. Got rank %d" % inputs_rank)

    outputs = self._cnn(inputs["observation"]["agent_obs"])

    if batched_inputs:
      outputs = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      outputs = jnp.reshape(outputs, [-1])  # [D]

    outputs = self._ff(outputs)
    action = jax.nn.one_hot(inputs["action"], num_classes=self.num_actions)
    combined_op = jnp.concatenate([outputs, action], axis=-1)
    return combined_op


class MeltingpotFE(hk.Module):

  def __init__(self, num_actions):
    super().__init__("meltingpot_features")
    self.num_actions = num_actions
    self._visual_torso = VisualFeatures()

  def __call__(self, inputs):
    # extract environment observation from the full observation object
    obs = inputs["observation"]

    # extract visual features form RGB observation
    ip_img = obs["RGB"].astype(jnp.float32) / 255
    vis_op = self._visual_torso(ip_img)

    # extract other observations
    inventory, ready_to_shoot = obs["INVENTORY"], obs["READY_TO_SHOOT"]

    # Do a one-hot embedding of the actions.
    action = jax.nn.one_hot(
        inputs["action"], num_classes=self.num_actions)  # [B, A]

    # Add dummy trailing dimensions to rewards if necessary.
    while ready_to_shoot.ndim < inventory.ndim:
      ready_to_shoot = jnp.expand_dims(ready_to_shoot, axis=-1)

    # check if the dimensions of all the tensors match
    # assert vis_op.ndim==inventory.ndim==ready_to_shoot.ndim==action.ndim

    # concatenate all the results
    combined_op = jnp.concatenate([vis_op, ready_to_shoot, inventory, action],
                                  axis=-1)

    return combined_op

class MeltingpotFECNNVis(hk.Module):

  def __init__(self, num_actions):
    super().__init__("meltingpot_features")
    self.num_actions = num_actions
    self._visual_torso = VisualFeaturesCNNVis()

  def __call__(self, inputs):
    # extract environment observation from the full observation object
    obs = inputs["observation"]

    # extract visual features form RGB observation
    ip_img = obs["RGB"].astype(jnp.float32) / 255
    vis_op, cnn_attn = self._visual_torso(ip_img)

    # extract other observations
    inventory, ready_to_shoot = obs["INVENTORY"], obs["READY_TO_SHOOT"]

    # Do a one-hot embedding of the actions.
    action = jax.nn.one_hot(
        inputs["action"], num_classes=self.num_actions)  # [B, A]

    # Add dummy trailing dimensions to rewards if necessary.
    while ready_to_shoot.ndim < inventory.ndim:
      ready_to_shoot = jnp.expand_dims(ready_to_shoot, axis=-1)

    # check if the dimensions of all the tensors match
    # assert vis_op.ndim==inventory.ndim==ready_to_shoot.ndim==action.ndim

    # concatenate all the results
    combined_op = jnp.concatenate([vis_op, ready_to_shoot, inventory, action],
                                  axis=-1)

    return combined_op, cnn_attn  # Return both the final output and the attention features

class VisualFeatures(hk.Module):
  """Simple convolutional stack from MeltingPot paper."""

  def __init__(self):
    super().__init__(name="meltingpot_visual_features")
    self._cnn = hk.Sequential([
        hk.Conv2D(16, [8, 8], 8, padding="VALID"),
        jax.nn.relu,
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

class VisualFeaturesCNNVis(hk.Module):
  """Simple convolutional stack from MeltingPot paper."""

  def __init__(self):
    super().__init__(name="meltingpot_visual_features")
    self._cnn_1 = hk.Sequential([
        hk.Conv2D(16, [8, 8], 8, padding="VALID"),
        jax.nn.relu,
    ])
    self._cnn_2 = hk.Sequential([
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

    cnn_attn = self._cnn_1(inputs)  # Shape: [B, 11, 11, 16]
    outputs = self._cnn_2(cnn_attn)  # Shape: [B, 8, 8, 32]

    # Reduce channel dimension of cnn_attn as sum of absolute values
    cnn_attn = jnp.sum(jnp.abs(cnn_attn), axis=-1)  # Shape: [B, 11, 11]
    cnn_attn = jnp.reshape(cnn_attn, (-1, cnn_attn.shape[-1] * cnn_attn.shape[-2]))  # [B, 121]
    cnn_attn = jax.nn.softmax(cnn_attn, axis=-1)    
    # To be consistent with multihead attention, reshape to [B, 1, n_heads=1, 121]
    cnn_attn = jnp.expand_dims(cnn_attn, axis=-2)  # [B, 1, 121]
    cnn_attn = jnp.expand_dims(cnn_attn, axis=-2)  # [B, 1, 1, 121]

    if batched_inputs:
      outputs = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      outputs = jnp.reshape(outputs, [-1])  # [D]

    outputs = self._ff(outputs)
    return outputs, cnn_attn  # Return both the final output and the attention features

class AttentionCNN_FE(hk.Module):

  def __init__(self, num_actions, flatten_output=True, num_channels = 64):
    super().__init__("meltingpot_features")
    self.num_actions = num_actions
    self._visual_torso = VisualFeatures_attention(flatten_output=flatten_output, num_channels=num_channels)

  def __call__(self, inputs):
    # extract environment observation from the full observation object
    obs = inputs["observation"]

    # extract visual features form RGB observation
    ip_img = obs["RGB"].astype(jnp.float32) / 255
    vis_op = self._visual_torso(ip_img)

    return vis_op
  
class AttentionCNN_FE_SelfSupervise(hk.Module):

  def __init__(self, num_actions, flatten_output=True, num_channels = 64):
    super().__init__("meltingpot_features")
    self.num_actions = num_actions
    self._visual_torso = VisualFeatures_attention_selfsupervise(num_channels=num_channels)

  def __call__(self, inputs):
    # extract environment observation from the full observation object
    obs = inputs["observation"]

    # extract visual features form RGB observation
    ip_img = obs["RGB"].astype(jnp.float32) / 255
    vis_op, guidance_map = self._visual_torso(ip_img)

    return vis_op, guidance_map

class AttentionSpatialCNN_FE(hk.Module):

  def __init__(self, num_actions, flatten_output=True, num_channels = 16):
    super().__init__("meltingpot_features")
    self.num_actions = num_actions
    self._visual_torso = VisualFeatures_attention(flatten_output=flatten_output, num_channels=num_channels)

  def __call__(self, inputs):
    # extract environment observation from the full observation object
    obs = inputs["observation"]

    # extract visual features form RGB observation
    ip_img = obs["RGB"].astype(jnp.float32) / 255
    vis_op = self._visual_torso(ip_img)

    return vis_op

class VisualFeatures_attention(hk.Module):
  """Simple convolutional stack from MeltingPot paper."""

  def __init__(self, flatten_output=True, num_channels=64):
    super().__init__(name="meltingpot_visual_features")
    self.flatten_output = flatten_output
    self._cnn = hk.Sequential([
        hk.Conv2D(num_channels, [8, 8], 8, padding="VALID"),
        jax.nn.relu,
    ])

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
        raise ValueError("Expected input BHWC or HWC. Got rank %d" % inputs_rank)

    # Process through CNN layers
    outputs = self._cnn(inputs)  # Shape: [B, 11, 11, 64]
    
    if self.flatten_output:
      if batched_inputs:
          # Reshape to [B, 121, 64] for attention
          outputs = jnp.reshape(outputs, [outputs.shape[0], -1, outputs.shape[-1]])
      else:
          # Handle unbatched case
          outputs = jnp.reshape(outputs, [-1, outputs.shape[-1]])  # [121, 64]

    return outputs

class VisualFeatures_attention_selfsupervise(hk.Module):
  """Visual CNN + self-supervised attention guidance.

  - Features are always flattened to [B, 121, 64].
  - Guidance map is shaped [B, 1, 11, 11].
  - Guidance is non-differentiable (no gradients traced).
  """

  def __init__(self, num_channels=64, match_atol=0.0, eps=1e-8):
    super().__init__(name="meltingpot_visual_features_selfsupervise")
    self.match_atol = match_atol
    self.eps = eps
    self._cnn = hk.Sequential([
        hk.Conv2D(num_channels, [8, 8], 8, padding="VALID"),  # [B, 11, 11, 64]
        jax.nn.relu,
    ])

  def __call__(self, inputs: jnp.ndarray):
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError(f"Expected input BHWC or HWC. Got rank {inputs_rank}")

    feats = self._cnn(inputs)  # [B, 11, 11, 64] or [11, 11, 64]

    if batched_inputs:
      outputs = jnp.reshape(feats, [feats.shape[0], -1, feats.shape[-1]])  # [B, 121, 64]
    else:
      outputs = jnp.reshape(feats, [-1, feats.shape[-1]])                  # [121, 64]

    # Build guidance WITHOUT tracing gradients from outputs.
    guidance = self._build_guidance(outputs, batched_inputs)  # [B,1,11,11] or [1,1,11,11]

    return outputs, guidance

  def _build_guidance(self, outputs, batched: bool):
      """
      Compute class-wise softmax(1/freq) (implemented as softmax(-log freq)),
      keep rarest classes up to 30% coverage, reshape to [B,1,11,11] or [1,1,11,11].
      Gradients are stopped on the returned guidance.
      """
      x = outputs  # we’ll stop grads at the very end

      def pair_eq(a):  # a: [M, D] or [B, M, D] with vmap
          # L∞ distance <= atol is equivalent to isclose with rtol=0
          return (jnp.max(jnp.abs(a[..., None, :, :] - a[..., :, None, :]), axis=-1) <= self.match_atol)

      def guidance_from_counts(counts):
          # counts: [M] or [B, M], integer >= 1
          M = counts.shape[-1]

          # Per-(batch) histogram with vmap over rows
          if counts.ndim == 2:
              hist = jax.vmap(lambda c: jnp.bincount(c, length=M + 1))(counts)   # [B, M+1]
              cum  = jnp.cumsum(hist, axis=-1)                                   # [B, M+1]
              K    = int(0.3 * M)
              allow_freq = cum <= K                                              # [B, M+1]
              kmin = jnp.min(counts, axis=-1)                                   # [B]
              allow_freq = jnp.logical_or(allow_freq, (jnp.arange(M+1)[None, :] == kmin[:, None]))
              mask_keep = jnp.take_along_axis(allow_freq[:, None, :], counts[..., None], axis=-1)[..., 0]  # [B, M]

          else:
              hist = jnp.bincount(counts, length=M + 1)                           # [M+1]
              cum  = jnp.cumsum(hist)                                             # [M+1]
              K    = int(0.3 * M)
              allow_freq = cum <= K   
              kmin = jnp.min(counts)
              allow_freq = jnp.logical_or(allow_freq, (jnp.arange(M+1) == kmin))                                            # [M+1]
              mask_keep  = allow_freq[counts]                                     # [M]

          fcounts = counts.astype(jnp.result_type(x, jnp.float32))
          logits  = jnp.where(mask_keep, -jnp.log(fcounts), -jnp.inf)             # same ordering as 1/freq
          return jax.nn.softmax(logits, axis=-1)

      if batched:
          # x: [B, 121, 64]
          eq = pair_eq(x)                         # [B, M, M]
          counts = jnp.sum(eq, axis=-1)           # [B, M]
          attn = guidance_from_counts(counts)     # [B, M]
          guidance = attn.reshape(attn.shape[0], 1, 11, 11)   # [B, 1, 11, 11]
      else:
          # x: [121, 64]
          eq = pair_eq(x)                         # [M, M]
          counts = jnp.sum(eq, axis=-1)           # [M]
          attn = guidance_from_counts(counts)     # [M]
          guidance = attn.reshape(1, 1, 11, 11)   # [1, 1, 11, 11]
          #jax.debug.print("Guidance coverage: {}/121", jnp.sum(guidance > 0.01))
      return jax.lax.stop_gradient(guidance)


  
class Discriminator(hk.Module):

  def __init__(self, diversity_dim, discriminator_ensembles):
    super().__init__(name="discriminator")
    self._diversity_dim = diversity_dim
    self._discriminator_ensembles = discriminator_ensembles
    self._layer = hk.Linear(diversity_dim * discriminator_ensembles)

  def __call__(self, inputs: jnp.array):
    op = self._layer(inputs)
    op = op.reshape((-1, self._discriminator_ensembles, self._diversity_dim))
    # op = jax.nn.softmax(op, axis=-1)
    return op
