import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def get_llm(model_name="google/gemini-2.0-flash-exp:free", temperature=0):
    """Create a ChatOpenAI client configured for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(f"OPENROUTER_API_KEY not found in {env_path}")

    data_collection = os.getenv("OPENROUTER_DATA_COLLECTION", "deny").strip().lower()
    if data_collection not in {"deny", "allow"}:
        data_collection = "deny"

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        model_kwargs={
            "extra_body": {
                "provider": {
                    "data_collection": data_collection,
                }
            }
        },
    )
