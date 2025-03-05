from acme.specs import EnvironmentSpec
from acme.tf import savers as tf_savers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from absl import flags
from absl import app
import wandb

import train
from marl import experiments
from marl import specs as ma_specs
from marl.experiments import config as ma_config
from marl.agents.networks import VisualFeatures

Images = jnp.ndarray # Useful type aliases
FLAGS = flags.FLAGS

class CNNProbe(hk.Module):

  def __init__(self, output_size: int):
    super().__init__("meltingpot_cnn_probe")
    self._visual_torso = VisualFeatures()
    self._linear = hk.Linear(output_size)

  def __call__(self, obs):
    # extract visual features form RGB observation
    ip_img = obs.astype(jnp.float32) / 255
    vis_op = self._visual_torso(ip_img)
    logits = self._linear(vis_op)
    return logits

def load_checkpoint(agent_idx: int = 0):
    """
    Loads a checkpoint dictionary using ACME's TF checkpointer.
    This checkpointer expects the directory structure that ACME learners produce.
    """

    experiment, experiment_dir = train.build_experiment_config()
    checkpointing_config = ma_config.CheckpointingConfig(
        max_to_keep=3, directory=experiment_dir, add_uid=False)

    # if FLAGS.agent_param_indices is not None:
    #     agent_param_indices = [int(idx) for idx in FLAGS.agent_param_indices.split(",")]
    #     experiment.agent_param_indices = agent_param_indices

    key = jax.random.PRNGKey(experiment.seed)

    # Create the environment and get its spec.
    environment = experiment.environment_factory(experiment.seed)
    environment_specs: ma_specs.MAEnvironmentSpec = experiment.environment_spec
    scenario_spec: ma_specs.MAEnvironmentSpec = ma_specs.MAEnvironmentSpec(
        environment)

    # Create the networks and policy.
    network = experiment.network_factory(
        environment_specs.get_single_agent_environment_specs())

    dataset = None  # fakes.transition_dataset_from_spec(environment_specs.get_agent_environment_specs())

    learner_key, key = jax.random.split(key)
    learner = experiment.builder.make_learner(
        random_key=learner_key,
        networks=network,
        dataset=dataset,
        logger_fn=experiment.logger_factory,
        environment_spec=environment_specs,
    )

    s0 = learner._combined_states.params.copy()
    checkpointer = tf_savers.Checkpointer(
        objects_to_save={"learner": learner},
        directory=checkpointing_config.directory,
        subdirectory="learner",
        time_delta_minutes=checkpointing_config.model_time_delta_minutes,
        add_uid=checkpointing_config.add_uid,
        max_to_keep=checkpointing_config.max_to_keep,
    )
    checkpointer.restore()
    s1 = learner._combined_states.params
            
    return s1

def replace_params(restored_params: dict, new_params: dict, agent_idx: int = 0):
    for layer in ['conv2_d', 'conv2_d_1', 'linear', 'linear_1']:
        for ele in ['w', 'b']:
            new_params[f'meltingpot_cnn_probe/~/meltingpot_visual_features/~/{layer}'][ele] = \
                restored_params[f'impala_network/~/meltingpot_features/~/meltingpot_visual_features/~/{layer}'][ele][agent_idx]
    return new_params

def build_standalone_model_from_checkpoint(transformed_forward: hk.Transformed, agent_idx: int = 0):
    # Load the checkpoint (parameters) from disk:
    restored = load_checkpoint()

    # Make a dummy input to initialize the new model's parameters.
    dummy_input = jnp.zeros((1, 64, 64, 3), dtype=jnp.float32)

    # Use hk.transform to initialize parameters in the new (CNN+Linear) model.
    rng = jax.random.PRNGKey(42)
    new_params = transformed_forward.init(rng, dummy_input)
   
    # Replace the parameters in new_params with the CNN parameters from the checkpoint:
    params = replace_params(restored, new_params, agent_idx)

    return params

def freeze_cnn_mask(params):
    """
    Create a mask pytree indicating which parameters belong to the CNN (False)
    and which belong to the linear head (True). The mask will be used by
    optax.multi_transform to decide which optimizer transform is applied.
    """
    def _mask_fn(module_name):
        return 'meltingpot_visual_features' not in module_name

    def _traverse(tree, prefix=''):
        if not isinstance(tree, dict):
            # Leaf param
            return _mask_fn(prefix)
        out = {}
        for k, v in tree.items():
            # Construct a module "prefix" we can test
            new_prefix = prefix + '/' + k if prefix else k
            out[k] = _traverse(v, new_prefix)
        return out

    return _traverse(params)

def create_linear_only_optimizer(learning_rate=1e-3):
    """
    Creates an Optax optimizer that:
      - Uses Adam on the linear parameters.
      - Sets CNN parameters' gradients to 0 (frozen).
    """
    # We define two transforms:
    #   "train": standard Adam update
    #   "freeze": sets grads to zero
    transforms = {
        'train': optax.adam(learning_rate),
        'freeze': optax.set_to_zero()
    }

    # param_mask: a pytree of the same structure as params,
    # whose leaves are either 'train' or 'freeze'
    def label_fn(mask_leaf):
        return 'train' if mask_leaf else 'freeze'

    return transforms, label_fn

def train_model(transformed_forward: hk.Transformed,
                init_params,
                rng,
                num_steps=1000,
                batch_size=32,
                lr=1e-3):
    """
    Example training loop that freezes the CNN params and trains only the linear head.
    Uses random data + a simple classification objective for demonstration.
    """

    # Create a mask: True => train, False => freeze
    mask = freeze_cnn_mask(init_params)

    # Create the multi_transform-based optimizer
    transforms, label_fn = create_linear_only_optimizer(learning_rate=lr)
    tx = optax.multi_transform(transforms, jax.tree_map(label_fn, mask))
    opt_state = tx.init(init_params)

    @jax.jit
    def loss_fn(params, batch, labels):
        logits = transformed_forward.apply(params, rng, batch)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss

    @jax.jit
    def update(params, opt_state, batch, labels):
        grads = jax.grad(loss_fn)(params, batch, labels)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    params = init_params

    for step in range(num_steps):
        # Generate random images and labels
        batch_rng, rng = jax.random.split(rng)
        fake_images = jax.random.normal(batch_rng, (batch_size, 88, 88, 3))
        label_rng, rng = jax.random.split(rng)
        fake_labels = jax.random.randint(label_rng, (batch_size,), 0, 2)

        params, opt_state = update(params, opt_state, fake_images, fake_labels)

        if step % 100 == 0:
            curr_loss = loss_fn(params, fake_images, fake_labels)
            print(f"Step {step}: loss = {curr_loss:.4f}")

            # 3) Log the current loss to wandb
            wandb.log({"train_loss": float(curr_loss), "step": step})

    return params

def main(_):
    wandb.init(project="cnn_probe",
               config={
                   "num_steps": 500,
                   "batch_size": 16,
                   "learning_rate": 1e-4
               })

    def forward_fn(x):
        """Haiku transform function that instantiates CNNProbe."""
        model = CNNProbe(output_size=2)
        return model(x)

    transformed_forward = hk.transform(forward_fn)

    # Build the model params with CNN loaded from checkpoint:
    new_params = build_standalone_model_from_checkpoint(
        transformed_forward, agent_idx=0
    )

    rng = jax.random.PRNGKey(999)
    
    trained_params = train_model(
        transformed_forward,
        init_params=new_params,
        rng=rng,
        num_steps=wandb.config.num_steps,
        batch_size=wandb.config.batch_size,
        lr=wandb.config.learning_rate
    )

    # Example inference with the newly trained parameters:
    sample_input = jax.random.normal(rng, (2, 88, 88, 3))
    logits = transformed_forward.apply(trained_params, rng, sample_input)
    print("Final Logits shape:", logits.shape)
    print("Final Logits sample:", logits)

    wandb.finish()

if __name__ == "__main__":
    app.run(main)