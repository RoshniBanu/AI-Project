import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from stable_baselines3 import PPO
from env.grid_world import GridWorldEnv
from rewards.reward_builder import validate_reward_config, build_reward_fn
from llm.reward_generator import generate_reward_config
from evaluation.evaluate import evaluate_policy_custom

# ---- Get objective from the user, live ----
print("=== LLM-Guided Reward Shaping Demo ===\n")
objective = input("Enter your task objective in plain English:\n> ").strip()

if not objective:
    objective = "Reach the target while avoiding obstacles and minimizing unnecessary movement."
    print(f"(No input given, using default: \"{objective}\")")

print(f"\nObjective: {objective}\n")
print("Calling LLM to generate reward config...")
llm_config = generate_reward_config(objective)
print(f"LLM proposed config: {llm_config}\n")

valid, msg = validate_reward_config(llm_config)
print(f"Validation result: {valid} - {msg}\n")

if not valid:
    raise ValueError(f"LLM config failed validation: {msg}")

reward_fn = build_reward_fn(llm_config)
env = GridWorldEnv(reward_fn=reward_fn)

print("Training PPO with LLM-generated reward...")
model = PPO("MlpPolicy", env, verbose=0)  # verbose=0 to keep demo output clean
model.learn(total_timesteps=20000)
print("Training complete.\n")

print("Evaluating...")
eval_env = GridWorldEnv(reward_fn=reward_fn)
report = evaluate_policy_custom(model, eval_env, n_episodes=50)

print("=== Evaluation Report ===")
for k, v in report.items():
    print(f"{k}: {v}")