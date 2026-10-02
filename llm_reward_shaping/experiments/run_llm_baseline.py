import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from stable_baselines3 import PPO
from env.grid_world import GridWorldEnv
from rewards.reward_builder import validate_reward_config, build_reward_fn
from llm.reward_generator import generate_reward_config
from evaluation.evaluate import evaluate_policy_custom

# ---- Baseline B: LLM-generated reward (no refinement) ----
objective = "Reach the target while avoiding obstacles and minimizing unnecessary movement."

print(f"Objective: {objective}\n")
print("Calling LLM to generate reward config...")
llm_config = generate_reward_config(objective)
print(f"LLM proposed config: {llm_config}\n")

valid, msg = validate_reward_config(llm_config)
print(f"Validation result: {valid} - {msg}\n")

if not valid:
    raise ValueError(f"LLM config failed validation: {msg}")

# Save this config for the record (you'll need it later for Module 7 reflection)
os.makedirs("results", exist_ok=True)
with open("results/llm_config_round1.json", "w") as f:
    json.dump(llm_config, f, indent=2)

reward_fn = build_reward_fn(llm_config)
env = GridWorldEnv(reward_fn=reward_fn)

print("Training PPO with LLM-generated reward...")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("results/ppo_baseline_llm_round1")
print("Training complete. Model saved to results/ppo_baseline_llm_round1\n")

print("Evaluating...")
eval_env = GridWorldEnv(reward_fn=reward_fn)
report = evaluate_policy_custom(model, eval_env, n_episodes=50)

with open("results/eval_llm_round1.json", "w") as f:
    json.dump(report, f, indent=2)

print("=== Baseline B (LLM Round 1) Evaluation ===")
for k, v in report.items():
    print(f"{k}: {v}")