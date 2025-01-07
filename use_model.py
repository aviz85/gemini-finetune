"""Module for interacting with a fine-tuned Gemini model."""

import os
import argparse
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def load_model(model_name: str) -> Any:
    """Load the trained model.
    
    Args:
        model_name: The name of the model to load
        
    Returns:
        A GenerativeModel instance
    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel(model_name=model_name)

def chat_with_model(model: Any, prompt: str) -> str:
    """Chat with the model."""
    if not prompt.strip():
        return "Please enter a message"
    response = model.generate_content(prompt)
    return response.text

def main() -> None:
    """Run the interactive chat loop."""
    parser = argparse.ArgumentParser(description='Chat with your trained model')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., tunedModels/my-model-name)')
    args = parser.parse_args()
    
    try:
        model = load_model(args.model)
        print("Chat with your model (type 'quit' to exit)")
        
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            
            try:
                response = chat_with_model(model, user_input)
                print(f"AI: {response}")
            except Exception as e:
                print(f"Error: {str(e)}")
                
    except Exception as e:
        print(f"Failed to load model: {str(e)}")

if __name__ == "__main__":
    main() 