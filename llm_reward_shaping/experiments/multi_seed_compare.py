import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.grid_world import GridWorldEnv
from rewards.reward_builder import validate_reward_config, build_reward_fn
from evaluation.evaluate import evaluate_policy_custom

# Edit these to compare whichever configs you want
CONDITIONS = {
    "Manual / LLM R1 (10, -5, -0.1)": {"goal": 10.0, "obstacle": -5.0, "efficiency": -0.1},
    "Refined (15, -5, -0.3)": {"goal": 15.0, "obstacle": -5.0, "efficiency": -0.3},
}
SEEDS = [0, 1, 2, 3, 4]
TIMESTEPS = 20000
EVAL_EPISODES = 100

results = {}

for name, config in CONDITIONS.items():
    valid, msg = validate_reward_config(config)
    if not valid:
        raise ValueError(f"{name}: {msg}")

    reward_fn = build_reward_fn(config)
    success, steps = [], []

    print(f"\n=== {name} ===")
    for seed in SEEDS:
        env = GridWorldEnv(reward_fn=reward_fn)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0)
        model.learn(total_timesteps=TIMESTEPS)

        np.random.seed(1000 + seed)  # fixed evaluation starts per seed
        report = evaluate_policy_custom(
            model, GridWorldEnv(reward_fn=reward_fn), n_episodes=EVAL_EPISODES
        )
        success.append(report["success_rate"])
        steps.append(report["avg_episode_length"])
        print(f"  seed {seed}: success={report['success_rate']:.2f}, steps={report['avg_episode_length']:.1f}")

    results[name] = {"success": success, "steps": steps}
    print(f"  MEAN success = {np.mean(success):.3f} ± {np.std(success):.3f}")
    print(f"  MEAN steps   = {np.mean(steps):.2f} ± {np.std(steps):.2f}")

os.makedirs("results", exist_ok=True)
with open("results/multi_seed_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Error-bar charts
names = list(results.keys())
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, key, title in [(axes[0], "success", "Success rate (mean ± std)"),
                       (axes[1], "steps", "Average steps (mean ± std, lower is better)")]:
    means = [np.mean(results[n][key]) for n in names]
    stds = [np.std(results[n][key]) for n in names]
    ax.bar(range(len(names)), means, yerr=stds, capsize=6,
           color=["#4C72B0", "#55A868", "#DD8452", "#8172B2"][:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_title(title)

plt.tight_layout()
plt.savefig("results/multi_seed_comparison.png")
print("\nSaved: results/multi_seed_results.json and results/multi_seed_comparison.png")