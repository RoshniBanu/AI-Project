import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from env.grid_world import GridWorldEnv
from rewards.reward_builder import validate_reward_config, build_reward_fn

# ---- Baseline A: Manual reward config ----
manual_config = {"goal": 10.0, "obstacle": -5.0, "efficiency": -0.1}

valid, msg = validate_reward_config(manual_config)
if not valid:
    raise ValueError(f"Invalid reward config: {msg}")

reward_fn = build_reward_fn(manual_config)

# Create environment with this reward function
env = GridWorldEnv(reward_fn=reward_fn)

# Train PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

# Save model with a name specific to this condition
model.save("results/ppo_baseline_manual")
print("Training complete. Model saved to results/ppo_baseline_manual")