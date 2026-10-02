import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from stable_baselines3 import PPO
from env.grid_world import GridWorldEnv
from rewards.reward_builder import validate_reward_config, build_reward_fn
from llm.reward_generator import refine_reward_config
from evaluation.evaluate import evaluate_policy_custom

# ---- Condition C: Refined LLM reward ----
objective = "Reach the target while avoiding obstacles and minimizing unnecessary movement."

with open("results/llm_config_round1.json") as f:
    previous_config = json.load(f)

with open("results/eval_llm_round1.json") as f:
    performance_report = json.load(f)

print(f"Previous config: {previous_config}")
print(f"Previous performance: {performance_report}\n")

print("Calling LLM to refine reward config...")
refined_config = refine_reward_config(objective, previous_config, performance_report)
print(f"LLM refined config: {refined_config}\n")

valid, msg = validate_reward_config(refined_config)
print(f"Validation result: {valid} - {msg}\n")

if not valid:
    raise ValueError(f"Refined config failed validation: {msg}")

with open("results/llm_config_round2.json", "w") as f:
    json.dump(refined_config, f, indent=2)

reward_fn = build_reward_fn(refined_config)
env = GridWorldEnv(reward_fn=reward_fn)

print("Training PPO with refined LLM reward...")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("results/ppo_condition_c_refined")
print("Training complete.\n")

print("Evaluating...")
eval_env = GridWorldEnv(reward_fn=reward_fn)
report = evaluate_policy_custom(model, eval_env, n_episodes=50)

with open("results/eval_condition_c.json", "w") as f:
    json.dump(report, f, indent=2)

print("=== Condition C (Refined LLM Reward) Evaluation ===")
for k, v in report.items():
    print(f"{k}: {v}")