import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import matplotlib.pyplot as plt

# Load all three evaluation reports
with open("results/eval_llm_round1.json") as f:
    eval_b = json.load(f)
with open("results/eval_condition_c.json") as f:
    eval_c = json.load(f)

# Baseline A results (hardcode from your earlier run, or save it to a file too)
eval_a = {
    "success_rate": 0.70,
    "avg_reward": 5.318,
    "avg_episode_length": 17.52,
    "avg_collisions": 0.0,
    "reward_variance": 45.649
}

conditions = ["Manual (A)", "LLM Round 1 (B)", "Refined LLM (C)"]
reports = [eval_a, eval_b, eval_c]

# ---- Table ----
print(f"{'Metric':<22}{'Manual (A)':<15}{'LLM R1 (B)':<15}{'Refined (C)':<15}")
for key in ["success_rate", "avg_reward", "avg_episode_length", "avg_collisions"]:
    row = f"{key:<22}"
    for r in reports:
        row += f"{r[key]:<15.3f}"
    print(row)

# ---- Bar chart: Success rate ----
success_rates = [r["success_rate"] for r in reports]
plt.figure(figsize=(6, 4))
plt.bar(conditions, success_rates, color=["#4C72B0", "#DD8452", "#55A868"])
plt.ylabel("Success Rate")
plt.title("Success Rate by Reward Condition")
plt.ylim(0, 1)
for i, v in enumerate(success_rates):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig("results/success_rate_comparison.png")
print("\nSaved: results/success_rate_comparison.png")

# ---- Bar chart: Episode length ----
lengths = [r["avg_episode_length"] for r in reports]
plt.figure(figsize=(6, 4))
plt.bar(conditions, lengths, color=["#4C72B0", "#DD8452", "#55A868"])
plt.ylabel("Average Episode Length (steps)")
plt.title("Episode Efficiency by Reward Condition")
for i, v in enumerate(lengths):
    plt.text(i, v + 0.3, f"{v:.1f}", ha="center")
plt.tight_layout()
plt.savefig("results/episode_length_comparison.png")
print("Saved: results/episode_length_comparison.png")

# ---- Bar chart: Avg reward ----
rewards = [r["avg_reward"] for r in reports]
plt.figure(figsize=(6, 4))
plt.bar(conditions, rewards, color=["#4C72B0", "#DD8452", "#55A868"])
plt.ylabel("Average Cumulative Reward")
plt.title("Average Reward by Condition (not directly comparable due to reward scale differences)")
for i, v in enumerate(rewards):
    plt.text(i, v + 0.2, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig("results/avg_reward_comparison.png")
print("Saved: results/avg_reward_comparison.png")