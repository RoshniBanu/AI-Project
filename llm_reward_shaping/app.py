import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import json
from stable_baselines3 import PPO
from env.grid_world import GridWorldEnv
from rewards.reward_builder import validate_reward_config, build_reward_fn
from llm.reward_generator import generate_reward_config
from evaluation.evaluate import evaluate_policy_custom
from evaluation.visualize_path import get_agent_path, plot_grid_path

st.set_page_config(page_title="LLM-Guided Reward Shaping", layout="wide")

st.title("🤖 LLM-Guided Adaptive Reward Shaping")
st.caption("A closed-loop framework for reinforcement learning agents")

# ---- Session state to persist results across reruns ----
if "config" not in st.session_state:
    st.session_state.config = None
if "model" not in st.session_state:
    st.session_state.model = None
if "report" not in st.session_state:
    st.session_state.report = None

# ---- Step 1: User objective input ----
st.header("1️⃣ Task Objective")
objective = st.text_area(
    "Describe what the agent should do, in plain English:",
    value="Reach the target while avoiding obstacles and minimizing unnecessary movement.",
    height=80
)

col1, col2 = st.columns(2)

# ---- Step 2: Generate reward via LLM ----
with col1:
    st.header("2️⃣ Generate Reward")
    if st.button("🧠 Ask LLM to Generate Reward", use_container_width=True):
        with st.spinner("Calling Gemini..."):
            config = generate_reward_config(objective)
            valid, msg = validate_reward_config(config)

        st.session_state.config = config
        st.session_state.model = None  # reset downstream state
        st.session_state.report = None

        if valid:
            st.success(f"Validation: {msg}")
        else:
            st.error(f"Validation failed: {msg}")

    if st.session_state.config:
        st.json(st.session_state.config)

# ---- Step 3: Train agent ----
with col2:
    st.header("3️⃣ Train & Evaluate")
    train_disabled = st.session_state.config is None
    if st.button("🏋️ Train PPO Agent", use_container_width=True, disabled=train_disabled):
        valid, msg = validate_reward_config(st.session_state.config)
        if not valid:
            st.error(f"Cannot train — invalid config: {msg}")
        else:
            reward_fn = build_reward_fn(st.session_state.config)
            env = GridWorldEnv(reward_fn=reward_fn)

            with st.spinner("Training PPO (20,000 timesteps)... this may take ~20-30 seconds"):
                model = PPO("MlpPolicy", env, verbose=0)
                model.learn(total_timesteps=20000)

            eval_env = GridWorldEnv(reward_fn=reward_fn)
            report = evaluate_policy_custom(model, eval_env, n_episodes=50)

            st.session_state.model = model
            st.session_state.report = report

    if st.session_state.report:
        st.subheader("📊 Performance")
        r = st.session_state.report
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Success Rate", f"{r['success_rate']*100:.0f}%")
        m2.metric("Avg Reward", f"{r['avg_reward']:.2f}")
        m3.metric("Avg Steps", f"{r['avg_episode_length']:.1f}")
        m4.metric("Collisions", f"{r['avg_collisions']:.1f}")
    if st.session_state.model:
        st.subheader("🗺️ Agent Path (sample episode)")
        reward_fn = build_reward_fn(st.session_state.config)
        viz_env = GridWorldEnv(reward_fn=reward_fn)
        path, success = get_agent_path(st.session_state.model, viz_env)
        fig = plot_grid_path(viz_env, path, success, title="Trained Agent")
        st.pyplot(fig)

st.divider()
st.caption("LLM-Guided Adaptive Reward Shaping — AI Project | Grid World + PPO + Gemini")