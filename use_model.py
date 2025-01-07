"""Module for interacting with a fine-tuned Gemini model."""

import os
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
    """Chat with the model.
    
    Args:
        model: The GenerativeModel instance
        prompt: The input text to send to the model
        
    Returns:
        The model's response text
    """
    response = model.generate_content(prompt)
    return response.text

def main() -> None:
    """Run the interactive chat loop."""
    model_name = "tunedModels/myassistant-ujdib5etcy2r"
    model = load_model(model_name)
    
    print("Chat with your model (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        response = chat_with_model(model, user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main() 