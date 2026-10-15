import os
import sys
import json
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from llm.prompt_templates import (
    build_reward_generation_prompt,
    build_reward_refinement_prompt,
)

load_dotenv()

# Order of providers to try. Override in .env, e.g. LLM_PROVIDERS=groq
PROVIDERS = [p.strip().lower() for p in os.getenv("LLM_PROVIDERS", "gemini,groq").split(",") if p.strip()]
GEMINI_MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-3.6-flash").split(",") if m.strip()]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # check console.groq.com for current names
print(f"[LLM config] providers={PROVIDERS}, groq_model={GROQ_MODEL}")

TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "overloaded", "rate limit")

_gemini_client = None
_groq_client = None


def _gemini_call(prompt, model_name):
    global _gemini_client
    from google import genai
    from google.genai import types
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = _gemini_client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return response.text


def _groq_call(prompt, model_name):
    global _groq_client
    from groq import Groq
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = _groq_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _candidates():
    """(provider, model) pairs to try, skipping providers with no API key."""
    pairs = []
    for provider in PROVIDERS:
        if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
            pairs += [("gemini", m) for m in GEMINI_MODELS]
        elif provider == "groq" and os.getenv("GROQ_API_KEY"):
            pairs.append(("groq", GROQ_MODEL))
    return pairs


def _call_llm(prompt: str, attempts_per_model: int = 2):
    raw_text = None
    last_error = None

    for provider, model_name in _candidates():
        for attempt in range(1, attempts_per_model + 1):
            try:
                if provider == "gemini":
                    raw_text = _gemini_call(prompt, model_name)
                else:
                    raw_text = _groq_call(prompt, model_name)
                break
            except Exception as e:
                last_error = e
                #if not any(m.lower() in str(e).lower() for m in TRANSIENT_MARKERS):
                 #   raise  # real error (bad key, bad request): don't hide it
                if not any(m.lower() in str(e).lower() for m in TRANSIENT_MARKERS):
                    print(f"[LLM] {provider}/{model_name}: non-retryable error ({str(e)[:100]}). Trying next option...")
                    break
                wait = 2 ** attempt
                print(f"[LLM] {provider}/{model_name}: temporary error (attempt {attempt}/{attempts_per_model}). Retrying in {wait}s...")
                time.sleep(wait)
        if raw_text is not None:
            print(f"[LLM] Succeeded with {provider}/{model_name}")
            break
        print(f"[LLM] {provider}/{model_name} unavailable, trying next option...")

    if raw_text is None:
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON.\nRaw output was:\n{raw_text}") from e

    reasoning = str(data.pop("reasoning", "No reasoning provided."))
    return data, reasoning


def generate_reward_with_reasoning(objective: str):
    return _call_llm(build_reward_generation_prompt(objective))


def refine_reward_with_reasoning(objective: str, previous_config: dict, performance_report: dict):
    return _call_llm(build_reward_refinement_prompt(objective, previous_config, performance_report))


# ---- Backward-compatible wrappers (return config only) ----
def generate_reward_config(objective: str) -> dict:
    config, _ = generate_reward_with_reasoning(objective)
    return config


def refine_reward_config(objective: str, previous_config: dict, performance_report: dict) -> dict:
    config, _ = refine_reward_with_reasoning(objective, previous_config, performance_report)
    return config


if __name__ == "__main__":
    config, reasoning = generate_reward_with_reasoning("Avoid obstacles at all costs even if it takes longer.")
    print("Config:", config)
    print("Reasoning:", reasoning)