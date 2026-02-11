from absl import flags

FLAGS = flags.FLAGS

# General configurations
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("available_gpus", "0", "Comma separated list of GPU ids.")
flags.DEFINE_integer(
    "num_actors",
    8,
    "Number of actors to use (should be less than total number of CPU cores).",
)
flags.DEFINE_integer("actors_per_node", 1, "Number of actors per thread.")
flags.DEFINE_bool("inference_server", False, "Whether to run inference server.")

# Training configurations
flags.DEFINE_integer("num_steps", None, "Number of env steps to run.")
flags.DEFINE_bool(
    "async_distributed",
    False,
    "Should an agent be executed in an off-policy distributed way",
)
flags.DEFINE_bool("run_eval", False, "Whether to run evaluation.")
flags.DEFINE_bool(
    "all_parallel",
    False,
    "Flag to run all agents in parallel using vmap. Only use if GPU with large memory is available.",
)
flags.DEFINE_string("frozen_agents", None, "Comma separated list of frozen agents.")

# Environment configurations
flags.DEFINE_enum(
    "env_name",
    "overcooked",
    ["meltingpot", "overcooked"],
    "Environment to train on",
)
flags.DEFINE_string(
    "map_name",
    "cramped_room",
    "Meltingpot/Overcooked Map to train on. Only used when 'env_name' is 'meltingpot' or 'overcooked'",
)
flags.DEFINE_string("agent_roles", None, "Comma separated list of agent roles.")
flags.DEFINE_integer("reward_scale", 1, "Reward scale factor.")
flags.DEFINE_bool(
    "prosocial", False, "Whether to use shared reward for prosocial training."
)
flags.DEFINE_integer("max_episode_length", None, "Max Number of steps per episode.")
flags.DEFINE_string("map_layout", None, "Custom map layout for meltingpot maps")

# Coop-Mining specific flags
flags.DEFINE_bool(
    "conservative_mine_beam",
    False,
    "Whether to use conservative mining beam that penalizes mining",
)
flags.DEFINE_float("mining_reward", 0, "negative reward for mining")
flags.DEFINE_float("iron_reward", 1, "reward for iron")
flags.DEFINE_float("gold_reward", 4, "reward for gold")
flags.DEFINE_bool(
    "dense_ore_regrow", False, "Whether to use a larger ore regrowth rate"
)
flags.DEFINE_float("iron_rate", 0.0003, "iron regrow")
flags.DEFINE_float("gold_rate", 0.0002, "gold regrow")

# Algorithm configurations
flags.DEFINE_enum(
    "algo_name",
    "IMPALA",
    [
        "IMPALA", 
        "PopArtIMPALA",
        "PopArtIMPALA_CNN_visualization", 
        "OPRE", 
        "PopArtOPRE", 
        "PopArtIMPALA_attention",
        "PopArtIMPALA_attention_tanh",
        "PopArtIMPALA_attention_spatial",
        "PopArtIMPALA_attention_item_aware",
        "PopArtIMPALA_attention_multihead",
        "PopArtIMPALA_attention_multihead_ff",
        "PopArtIMPALA_attention_multihead_gated",
        "PopArtIMPALA_attention_multihead_disturb",
        "PopArtIMPALA_attention_multihead_enhance",
        "PopArtIMPALA_attention_multihead_item_aware",
        "PopArtIMPALA_attention_multihead_self_supervision",
        "simple_transformer",
        "simple_transformer_attention",
        "simple_transformer_cnnfeedback"
    ],
    "Algorithm to train",
)

# Attention network flags
flags.DEFINE_string("positional_embedding", None, "Whether to use positional embedding for attention")
flags.DEFINE_string("add_selection_vector", None, "Whether to add selection vector on the query in attention network")
flags.DEFINE_float("attn_enhance_multiplier", 0, "Attention enhancement multiplier")
flags.DEFINE_float("hidden_scale", 0, "ratio of cross attention")
flags.DEFINE_string("attn_enhance_head_indices", "0", "Comma separated list of attention heads to enhance.")
flags.DEFINE_string("attn_enhance_agent_skip_indices", "", "Comma separated list of agent indices to skip for attention enhancement.")
flags.DEFINE_integer("attn_enhance_item_idx", 0, "Index of the item to enhance attention on.")
flags.DEFINE_integer("attn_key_size", 64, "Size of the attention key vector.")
flags.DEFINE_string(
    "disturb_heads", "0",
    "Comma separated list of attention heads to disturb.",
)
flags.DEFINE_integer(
    "num_heads", 4, "Number of attention heads to use in the attention network."
)

# Loss selection flags
flags.DEFINE_float(
    "head_entropy_cost", 0.0, "Head entropy cost for attention networks."
)
flags.DEFINE_float(
    "attn_entropy_cost", 0.0, "attention entropy cost within each head."
)
flags.DEFINE_float(
    "head_cross_entropy_cost", 0.0, "Head cross entropy cost for attention networks."
)
flags.DEFINE_float(
    "head_mse_cost", 0.0, "Head MSE cost for attention networks."
)
flags.DEFINE_float(
    "reward_pred_cost", 0.0, "reward prediction cost for attention output."
)

# General logging and checkpointing configurations
flags.DEFINE_string("experiment_dir", None, "Directory to resume experiment from.")
flags.DEFINE_bool(
    "record_video", False, "Whether to record videos. (Only use during evaluation)"
)
flags.DEFINE_string(
    "exp_log_dir", "./results/", "Directory to store experiment logs in."
)
flags.DEFINE_bool("use_tb", False, "Flag to enable tensorboard logging.")
flags.DEFINE_bool("use_wandb", False, "Flag to enable wandb.ai logging.")
flags.DEFINE_string("wandb_entity", "linfangu-ucla", "Entity name for wandb account.")
flags.DEFINE_string("wandb_project", "marl-jax", "Project name for wandb logging.")
flags.DEFINE_string("wandb_tags", "", "Comma separated list of tags for wandb.")

# Observation logging flags
flags.DEFINE_bool("log_obs", False, "Whether to log observations.")
flags.DEFINE_string("log_filename", "temp/observations.jsonl", "Filename to log observations.")
flags.DEFINE_string(
    "log_img_dir", "agent_view_images", "Directory to save agent view images."
)
flags.DEFINE_integer("log_interval", 1, "Interval to log observations.")