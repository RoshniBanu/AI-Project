import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from stable_baselines3 import PPO

from env.grid_world import GridWorldEnv
from rewards.reward_builder import validate_reward_config, build_reward_fn, REWARD_SCHEMA
from llm.reward_generator import (
    generate_reward_with_reasoning,
    refine_reward_with_reasoning,
)
from evaluation.evaluate import evaluate_policy_custom
from evaluation.visualize_path import get_agent_path, plot_grid_path

MANUAL_CONFIG = {"goal": 10.0, "obstacle": -5.0, "efficiency": -0.1}
TIMESTEPS = 20000
MAX_ROUNDS = 4
PATH_SEED = 7
DEFAULT_OBJECTIVE = "Reach the target while avoiding obstacles and minimizing unnecessary movement."

st.set_page_config(page_title="LLM-Guided Reward Shaping", layout="wide")


def train_and_evaluate(config):
    """Trains PPO on the given reward config, evaluates it, and records a sample path."""
    reward_fn = build_reward_fn(config)

    env = GridWorldEnv(reward_fn=reward_fn)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)

    report = evaluate_policy_custom(model, GridWorldEnv(reward_fn=reward_fn), n_episodes=50)

    # Sample path from a fixed start so different rounds are visually comparable
    viz_env = GridWorldEnv(reward_fn=reward_fn)
    saved_state = np.random.get_state()
    np.random.seed(PATH_SEED)
    path, success = get_agent_path(model, viz_env)
    np.random.set_state(saved_state)

    return report, path, success


# ---------------- Session state ----------------
if "rounds" not in st.session_state:
    st.session_state.rounds = []
if "manual" not in st.session_state:
    st.session_state.manual = None
if "objective_used" not in st.session_state:
    st.session_state.objective_used = None

# ---------------- Sidebar: environment context (Stage 1 of the diagram) ----------------
with st.sidebar:
    st.header("🌍 Environment Context")
    ctx_env = GridWorldEnv()
    st.write(f"**Grid size:** {ctx_env.grid_size} × {ctx_env.grid_size}")
    st.write("**Start:** random free cell each episode")
    st.write(f"**Goal:** {ctx_env.goal}")
    st.write(f"**Obstacles:** {ctx_env.obstacles}")
    st.write("**Actions:** up, down, left, right")
    st.write(f"**Step limit:** {ctx_env.max_steps}")
    st.divider()
    st.header("🛡️ Validator bounds")
    st.caption("Any LLM output outside these ranges is rejected.")
    st.json(REWARD_SCHEMA)

# ---------------- Header ----------------
st.title("🤖 LLM-Guided Adaptive Reward Shaping")
st.caption("A closed-loop framework: LLM designs the reward → PPO learns → LLM reflects and refines")

# ---------------- Stage 1: objective ----------------
st.header("1️⃣ Task Objective")
objective = st.text_area(
    "Describe what the agent should do, in plain English:",
    value=DEFAULT_OBJECTIVE,
    height=80,
)

rounds = st.session_state.rounds
latest = rounds[-1] if rounds else None

# ---------------- Stages 2, 4, 6: action buttons ----------------
st.header("2️⃣ Pipeline Controls")
b1, b2, b3 = st.columns(3)

with b1:
    if st.button("🧠 1. Generate reward (LLM)", use_container_width=True):
        try:
            with st.spinner("Calling Gemini..."):
                config, reasoning = generate_reward_with_reasoning(objective)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
        else:
            valid, msg = validate_reward_config(config)
            if valid:
                st.session_state.rounds = [{
                    "label": "LLM Round 1",
                    "config": config,
                    "reasoning": reasoning,
                    "report": None,
                    "path": None,
                    "success": None,
                }]
                st.session_state.objective_used = objective
                st.rerun()
            else:
                st.error(f"Validator rejected the LLM output: {msg}")

with b2:
    can_train = latest is not None and latest["report"] is None
    if st.button("🏋️ 2. Train PPO agent", use_container_width=True, disabled=not can_train):
        with st.spinner(f"Training PPO ({TIMESTEPS:,} timesteps) and evaluating..."):
            report, path, success = train_and_evaluate(latest["config"])
        latest["report"] = report
        latest["path"] = path
        latest["success"] = success
        st.rerun()

with b3:
    can_refine = (
        latest is not None
        and latest["report"] is not None
        and len(rounds) < MAX_ROUNDS
    )
    if st.button("🔄 3. Refine reward & retrain", use_container_width=True, disabled=not can_refine):
        try:
            with st.spinner("Gemini is analysing the agent's performance..."):
                new_cfg, new_reason = refine_reward_with_reasoning(
                    st.session_state.objective_used, latest["config"], latest["report"]
                )
        except Exception as e:
            st.error(f"LLM call failed: {e}")
        else:
            valid, msg = validate_reward_config(new_cfg)
            if not valid:
                st.error(f"Validator rejected the refined config: {msg}")
            else:
                with st.spinner("Retraining on the refined reward..."):
                    report, path, success = train_and_evaluate(new_cfg)
                rounds.append({
                    "label": f"Refined Round {len(rounds) + 1}",
                    "config": new_cfg,
                    "reasoning": new_reason,
                    "report": report,
                    "path": path,
                    "success": success,
                })
                st.rerun()

if latest is not None and len(rounds) >= MAX_ROUNDS:
    st.caption(f"Maximum of {MAX_ROUNDS} rounds reached.")

# ---------------- Per-round results ----------------
if rounds:
    st.header("3️⃣ Rounds")
    for i, r in enumerate(rounds):
        with st.expander(r["label"], expanded=(i == len(rounds) - 1)):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Reward configuration**")
                st.json(r["config"])
                st.info(f"💡 **LLM reasoning:** {r['reasoning']}")

            with c2:
                st.markdown("**Performance**")
                if r["report"]:
                    rep = r["report"]
                    st.metric("Success rate", f"{rep['success_rate'] * 100:.0f}%")
                    st.metric("Avg steps to finish", f"{rep['avg_episode_length']:.1f}")
                    st.metric("Avg collisions", f"{rep['avg_collisions']:.2f}")
                    st.metric("Avg reward", f"{rep['avg_reward']:.2f}")
                else:
                    st.caption("Not trained yet. Click 'Train PPO agent'.")

            with c3:
                st.markdown("**Sample path (fixed start)**")
                if r["path"]:
                    fig = plot_grid_path(GridWorldEnv(), r["path"], r["success"], title=r["label"])
                    st.pyplot(fig)
                    plt.close(fig)

# ---------------- Comparison ----------------
st.header("4️⃣ Comparison")
st.caption(
    "Train the manual baseline (hand-written reward, no LLM) to compare against the LLM rounds. "
    "Manual config: " + str(MANUAL_CONFIG)
)

if st.button("📏 Train manual baseline (A)"):
    with st.spinner("Training manual baseline..."):
        report, path, success = train_and_evaluate(MANUAL_CONFIG)
    st.session_state.manual = {"report": report, "path": path, "success": success}
    st.rerun()

rows = []
if st.session_state.manual:
    rows.append(("Manual (A)", st.session_state.manual["report"]))
for r in rounds:
    if r["report"]:
        rows.append((r["label"], r["report"]))

if rows:
    df = pd.DataFrame([
        {
            "Condition": label,
            "Success rate": rep["success_rate"],
            "Avg steps": rep["avg_episode_length"],
            "Avg collisions": rep["avg_collisions"],
            "Avg reward": rep["avg_reward"],
        }
        for label, rep in rows
    ]).set_index("Condition")

    st.dataframe(df.round(3), use_container_width=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Success rate**")
        st.bar_chart(df[["Success rate"]])
    with ch2:
        st.markdown("**Average steps (lower is better)**")
        st.bar_chart(df[["Avg steps"]])

    st.caption(
        "Note: average reward is not comparable across conditions, because the LLM changes the reward "
        "scale each round. Compare success rate and steps instead. Each condition is a single training "
        "run, so small differences can be seed noise."
    )

st.divider()
st.caption("LLM-Guided Adaptive Reward Shaping | Grid World + PPO + Gemini")