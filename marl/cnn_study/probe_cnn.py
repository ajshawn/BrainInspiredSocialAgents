from acme.specs import EnvironmentSpec
from acme.tf import savers as tf_savers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from absl import flags
from absl import app
import wandb
from tqdm import tqdm

import train
from marl import experiments
from marl import specs as ma_specs
from marl.experiments import config as ma_config
from marl.agents.networks import VisualFeatures
from marl.cnn_study.make_dataset import build_tf_dataset

FLAGS = flags.FLAGS

flags.DEFINE_integer("ckp_idx", 0, "Checkpoint index loaded")
flags.DEFINE_integer("agent_idx", 0, "Agent index for which to train the probe")
flags.DEFINE_bool("random_baseline", False, "Probe a random CNN baseline")

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

def build_standalone_model_from_checkpoint(transformed_forward: hk.Transformed, agent_idx: int = 0, random_baseline=False):
    # Load the checkpoint (parameters) from disk:
    restored = load_checkpoint()

    # Make a dummy input to initialize the new model's parameters.
    dummy_input = jnp.zeros((1, 88, 88, 3), dtype=jnp.float32)

    # Use hk.transform to initialize parameters in the new (CNN+Linear) model.
    rng = jax.random.PRNGKey(42)
    new_params = transformed_forward.init(rng, dummy_input)
   
    # Replace the parameters in new_params with the CNN parameters from the checkpoint:
    if random_baseline:
        params = new_params
    else:
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

def train_binary_cls_model(
    transformed_forward: hk.Transformed,
    init_params,
    rng,
    dataset_config,
    num_steps=2000,
    validation_interval=100,
    verbose=False
):
    """
    Freezes the CNN params, trains only the linear head using the
    provided train_ds (training set) and val_ds (validation set).
    """
    # Create param mask & multi-transform
    mask = freeze_cnn_mask(init_params)
    transforms, label_fn = create_linear_only_optimizer(learning_rate=1e-4)
    tx = optax.multi_transform(transforms, jax.tree_map(label_fn, mask))
    opt_state = tx.init(init_params)

    @jax.jit
    def loss_fn(params, batch_images, batch_labels):
        logits = transformed_forward.apply(params, rng, batch_images)
        num_classes = logits.shape[-1]
        one_hot = jax.nn.one_hot(batch_labels, num_classes)
        return optax.softmax_cross_entropy(logits, one_hot).mean()

    @jax.jit
    def update(params, opt_state, batch_images, batch_labels):
        grads = jax.grad(loss_fn)(params, batch_images, batch_labels)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    params = init_params
    
    # Build the dataset
    train_ds, val_ds = build_tf_dataset(**dataset_config)

    for step in tqdm(range(num_steps)):
        # -- TRAINING --
        try:
            batch_images, batch_labels = next(train_ds)
        except StopIteration:
            # reset the iterator
            train_ds, _ = build_tf_dataset(**dataset_config)
            batch_images, batch_labels = next(train_ds)

        # Convert from tf-style float32 to JAX's jnp.float32
        batch_images = jnp.array(batch_images)
        batch_labels = jnp.array(batch_labels)

        params, opt_state = update(params, opt_state, batch_images, batch_labels)
        train_loss_val = loss_fn(params, batch_images, batch_labels)

        # Validation
        if step % validation_interval == 0:
            # Re-build or re-initialize val_ds iterator for a fresh pass
            _, val_ds = build_tf_dataset(**dataset_config)

            sum_val_loss = 0.0
            sum_correct = 0
            sum_count = 0

            num_classes = 2 # for binary classification
            correct_per_class = [0] * num_classes
            total_per_class = [0] * num_classes

            while True:
                try:
                    val_images, val_labels = next(val_ds)
                except StopIteration:
                    # We've reached the end of the validation iterator
                    break

                val_images = jnp.array(val_images)
                val_labels = jnp.array(val_labels)

                batch_val_loss = loss_fn(params, val_images, val_labels)
                sum_val_loss += float(batch_val_loss) * val_images.shape[0]

                # Predictions
                val_logits = transformed_forward.apply(params, rng, val_images)
                val_preds = jnp.argmax(val_logits, axis=-1)

                # Convert to numpy for easy counting
                val_preds_np = jnp.array(val_preds).astype(int)
                val_labels_np = jnp.array(val_labels).astype(int)

                # Update total correct & total count
                matches = (val_preds_np == val_labels_np)
                sum_correct += int(matches.sum())
                sum_count += val_images.shape[0]

                # ---- PER-CLASS STATS ----
                for i in range(val_images.shape[0]):
                    label_i = val_labels_np[i]
                    pred_i  = val_preds_np[i]
                    total_per_class[label_i] += 1
                    if label_i == pred_i:
                        correct_per_class[label_i] += 1

            # Compute average loss and accuracy over *all* validation batches
            if sum_count > 0:
                val_loss_avg = sum_val_loss / sum_count
                val_acc_avg = sum_correct / sum_count
            else:
                val_loss_avg = 0.0
                val_acc_avg = 0.0

            # Per-class accuracy
            per_class_acc = []
            for class_idx in range(num_classes):
                if total_per_class[class_idx] > 0:
                    acc = correct_per_class[class_idx] / total_per_class[class_idx]
                else:
                    acc = 0.0
                per_class_acc.append(acc)

            # Print summary
            if verbose:
                print(
                    f"Step {step}: "
                    f"train_loss={train_loss_val:.4f}, "
                    f"val_loss={val_loss_avg:.4f}, "
                    f"val_acc={val_acc_avg:.4f}, "
                    f"val_acc_per_class={per_class_acc}"
                )

            # WandB logging
            # You can log per-class accuracy in whatever format you prefer.
            wandb_dict = {
                "step": step,
                "train_loss": float(train_loss_val),
                "val_loss": float(val_loss_avg),
                "val_accuracy": float(val_acc_avg),
            }
            for class_idx, class_acc in enumerate(per_class_acc):
                wandb_dict[f"val_accuracy_class_{class_idx}"] = float(class_acc)

            wandb.log(wandb_dict)

    return params

def main(_):
    for label_key in ["apple", "floorAcorn", "preys", "predators"]:
    
        env_name = "predator_prey__open"
        log_name = f"{env_name}_{label_key}_agent{FLAGS.agent_idx}_ckpt{FLAGS.ckp_idx}" \
            if not FLAGS.random_baseline else f"{env_name}_{label_key}_random_baseline"
        wandb.init(
            project="cnn_probe",
            name=log_name,
            config={
                "num_steps": 5000,
                "batch_size": 16,
                "learning_rate": 1e-4,
                "agent_idx": FLAGS.agent_idx,
            }
        )
        
        dataset_config = {
            "label_key": label_key,
            "image_dir": "data/predator_prey__open_random/agent_view_images",
            "label_file": "data/predator_prey__open_random/observations.jsonl",
            "batch_size": wandb.config.batch_size,
            "task": "binary_cls",
            "num_epochs": None,
            "shuffle": False,
            "step_interval": 50,
            "n_agents": 13,
            "split_ratio": 0.8
        }

        # Build the forward
        def forward_fn(x):
            model = CNNProbe(output_size=2)  # 2-class classification
            return model(x)
        transformed_forward = hk.transform(forward_fn)

        # Load CNN parameters from checkpoint
        new_params = build_standalone_model_from_checkpoint(
            transformed_forward,
            agent_idx=FLAGS.agent_idx,
            random_baseline=FLAGS.random_baseline
        )

        rng = jax.random.PRNGKey(999)

        # Train the new linear head with the real dataset
        trained_params = train_binary_cls_model(
            transformed_forward=transformed_forward,
            init_params=new_params,
            dataset_config=dataset_config,
            rng=rng,
            num_steps=wandb.config.num_steps
        )

        wandb.finish()

if __name__ == "__main__":
    app.run(main)