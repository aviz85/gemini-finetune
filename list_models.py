"""Module for listing available fine-tuned models."""

import os
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def list_available_models() -> list[Any]:
    """List all available tuned models.
    
    Returns:
        List of model objects
    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.list_tuned_models()

def main() -> None:
    """Print all available models."""
    models = list_available_models()
    print("\nAvailable Models:")
    for model in models:
        print(f"- tunedModels/{model.name}")

if __name__ == "__main__":
    main() 