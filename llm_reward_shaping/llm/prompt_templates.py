import json

REWARD_GENERATION_PROMPT = """You are a reward design expert for reinforcement learning.

TASK OBJECTIVE (from user, in natural language):
"{objective}"

Your job: propose a reward configuration for a Grid World RL agent, using EXACTLY these three components:
- "goal": reward given when the agent reaches the target. Must be a positive number between 0 and 20.
- "obstacle": penalty given when the agent collides with an obstacle. Must be a negative number between -20 and 0.
- "efficiency": small penalty given at every step to encourage shorter paths. Must be a negative number between -2 and 0.

Also include a "reasoning" field: 2-3 plain sentences explaining why you chose these weights for this objective.

Respond with ONLY a valid JSON object in this exact format, with no extra text and no markdown:
{{"reasoning": "<your explanation>", "goal": <number>, "obstacle": <number>, "efficiency": <number>}}
"""

REWARD_REFINEMENT_PROMPT = """You are a reward design expert for reinforcement learning.

ORIGINAL TASK OBJECTIVE:
"{objective}"

PREVIOUS REWARD CONFIGURATION:
{previous_config}

OBSERVED AGENT PERFORMANCE AFTER TRAINING WITH THIS REWARD:
- Success rate: {success_rate}
- Average reward: {avg_reward}
- Average episode length: {avg_episode_length}
- Average collisions per episode: {avg_collisions}
- Reward variance (stability indicator): {reward_variance}

Analyze this performance and diagnose likely causes of any weaknesses (for example, low success rate,
long episodes, or high collisions can indicate an imbalance between the goal reward, obstacle penalty,
and step penalty).

Then propose a REVISED reward configuration, using EXACTLY these three components:
- "goal": positive number between 0 and 20
- "obstacle": negative number between -20 and 0
- "efficiency": negative number between -2 and 0

Also include a "reasoning" field: 2-3 plain sentences stating what you diagnosed from the metrics and
which weights you changed and why.

Respond with ONLY a valid JSON object in this exact format, with no extra text and no markdown:
{{"reasoning": "<your diagnosis and changes>", "goal": <number>, "obstacle": <number>, "efficiency": <number>}}
"""


def build_reward_generation_prompt(objective: str) -> str:
    return REWARD_GENERATION_PROMPT.format(objective=objective)


def build_reward_refinement_prompt(objective, previous_config, performance_report):
    return REWARD_REFINEMENT_PROMPT.format(
        objective=objective,
        previous_config=json.dumps(previous_config),
        success_rate=performance_report["success_rate"],
        avg_reward=performance_report["avg_reward"],
        avg_episode_length=performance_report["avg_episode_length"],
        avg_collisions=performance_report["avg_collisions"],
        reward_variance=performance_report["reward_variance"],
    )