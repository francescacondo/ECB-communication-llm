from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found in .env"
    )

client = OpenAI(api_key=api_key)

print("API key loaded successfully.")

response = client.responses.create(
    model="gpt-5-mini",
    input="Reply with exactly: API connection successful"
)

print("OpenAI response:")
print(response.output_text)