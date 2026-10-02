import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from google import genai
from llm.prompt_templates import build_reward_generation_prompt
from llm.prompt_templates import build_reward_refinement_prompt

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def generate_reward_config(objective: str) -> dict:
    prompt = build_reward_generation_prompt(objective)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    raw_text = response.text.strip()

    # Clean up in case the model wraps output in markdown code fences
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        config = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON.\nRaw output was:\n{raw_text}") from e

    return config


if __name__ == "__main__":
    objective = "Reach the target while avoiding obstacles and minimizing unnecessary movement."
    config = generate_reward_config(objective)
    print("Raw LLM-generated reward config:")
    print(config)


def refine_reward_config(objective: str, previous_config: dict, performance_report: dict) -> dict:
    prompt = build_reward_refinement_prompt(objective, previous_config, performance_report)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        config = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM refinement output as JSON.\nRaw output was:\n{raw_text}") from e

    return config