import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

N_TOTAL_AGENTS = 13
window_size = 20           # moving-average window
MILESTONE_EPISODES = [
    400654, 590233, 676868, 764833,
    832356, 833149, 983190, 1057156
]

prefix = (
    "results/predator_prey__open_1B_step/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323_ckp10684"
    "/csv_logs"
)
csv_file = f"{prefix}/actor.csv"

df = pd.read_csv(csv_file)

# Remove rows that are the repeated header written by the logger
df = df[~df["actor_episodes"].astype(str).str.contains("actor_episodes", na=False)]

# Make sure actor_episodes is numeric so we can compare it
df["actor_episodes"] = pd.to_numeric(df["actor_episodes"], errors="coerce")

for agent_idx in range(N_TOTAL_AGENTS):
    # numeric and aligned mask
    y_raw = pd.to_numeric(
        df[f"agent_{agent_idx}/episode_return"], errors="coerce"
    )
    mask = y_raw.notna() & df["actor_episodes"].notna()

    # X = running index we will plot on; Y = rewards
    Y = y_raw[mask].reset_index(drop=True)
    X = np.arange(len(Y))

    # positions in X where milestones occur
    episodes_aligned = df.loc[mask, "actor_episodes"].astype(int).values
    milestone_positions = [
        int(np.where(episodes_aligned == ep)[0][0])
        for ep in MILESTONE_EPISODES
        if ep in episodes_aligned
    ]

    # smoothed curve
    Y_smoothed = Y.rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(24, 8))
    plt.plot(X, Y, label="Original", alpha=0.3)
    plt.plot(X, Y_smoothed, label="Moving Average", linewidth=2)

    # draw vertical dashed red lines at the milestones
    for pos in milestone_positions:
        plt.axvline(x=pos, linestyle="--", color="red", alpha=0.7)

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"Agent {agent_idx} Reward (Smoothed)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{prefix}/agent_{agent_idx}_reward_smoothed_w_milestones.png")
    plt.clf()
