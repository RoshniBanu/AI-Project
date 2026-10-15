import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1) Every model your key can see, with no filtering, flash-style names only
print("=== Models visible to your key (names containing 'flash') ===")
for m in client.models.list():
    if "flash" in m.name.lower():
        print(m.name)

# 2) Try each candidate with a tiny request
candidates = [
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

print("\n=== Live test (tiny request to each) ===")
for name in candidates:
    try:
        r = client.models.generate_content(model=name, contents="Reply with the single word: ok")
        print(f"OK    {name}  ->  {r.text.strip()[:20]}")
    except Exception as e:
        print(f"FAIL  {name}  ->  {str(e)[:110]}")