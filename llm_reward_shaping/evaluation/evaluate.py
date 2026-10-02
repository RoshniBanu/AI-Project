import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from stable_baselines3 import PPO
from env.grid_world import GridWorldEnv
from rewards.reward_builder import build_reward_fn

def evaluate_policy_custom(model, env, n_episodes=50):
    success_count = 0
    episode_lengths = []
    episode_rewards = []
    collision_counts = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_length = 0
        ep_collisions = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            if info.get("collided", False):
                ep_collisions += 1

        success_count += 1 if done else 0
        episode_lengths.append(ep_length)
        episode_rewards.append(ep_reward)
        collision_counts.append(ep_collisions)

    report = {
        "success_rate": success_count / n_episodes,
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "avg_collisions": float(np.mean(collision_counts)),
        "reward_variance": float(np.var(episode_rewards)),
        "n_episodes": n_episodes,
    }
    return report


if __name__ == "__main__":
    manual_config = {"goal": 10.0, "obstacle": -5.0, "efficiency": -0.1}
    reward_fn = build_reward_fn(manual_config)
    env = GridWorldEnv(reward_fn=reward_fn)

    model = PPO.load("results/ppo_baseline_manual")
    report = evaluate_policy_custom(model, env, n_episodes=50)

    print("=== Baseline A (Manual Reward) Evaluation ===")
    for k, v in report.items():
        print(f"{k}: {v}")