import google.generativeai as genai
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

def get_default_examples() -> List[Dict[str, str]]:
    """Default training examples if no dataset file exists"""
    return [
        {"text_input": "מה שמך?", "output": "שמי הוא עוזר AI"},
        {"text_input": "מה אתה יכול לעשות?", "output": "אני יכול לעזור לך במגוון משימות כמו כתיבה, תכנות, וניתוח מידע"},
        {"text_input": "איך אתה יכול לעזור לי?", "output": "אני יכול לסייע בכתיבת קוד, פתרון בעיות תכנות, ניתוח נתונים, ומענה על שאלות"},
    ]

def create_training_examples() -> List[Dict[str, str]]:
    """Load examples from dataset.json or use defaults"""
    try:
        with open('dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            examples = data['training_examples']
            print(f"Found dataset.json with {len(examples)} training examples")
            return examples
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        default_examples = get_default_examples()
        print(f"Using {len(default_examples)} default training examples...")
        return default_examples

def load_config() -> Dict:
    """Load config from config.json or use defaults"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            print("Found config.json")
            return config['training_params']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        print("Using default config...")
        return {
            "display_name": "my-assistant",
            "epoch_count": 20,
            "batch_size": 2,
            "learning_rate": 0.001,
            "source_model": "models/gemini-1.5-flash-001-tuning"
        }

def train_model(api_key: str) -> genai.GenerativeModel:
    """Train a new model"""
    
    config = load_config()
    genai.configure(api_key=api_key)
    training_data = create_training_examples()
    
    print("Starting model training...")
    operation = genai.create_tuned_model(
        display_name=config["display_name"],
        source_model=config["source_model"],
        epoch_count=config["epoch_count"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        training_data=training_data
    )

    try:
        for status in operation.wait_bar():
            time.sleep(30)
            
        result = operation.result()
        print(f"\nTraining completed successfully!")
        print(f"Model name: {result.name}")
        
        # Plot loss function
        snapshots = pd.DataFrame(result.tuning_task.snapshots)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=snapshots, x='epoch', y='mean_loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Loss')
        plt.savefig('training_loss.png')
        plt.show()
        
        return genai.GenerativeModel(model_name=result.name)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def test_model(model: genai.GenerativeModel, test_input: str) -> str:
    """Test the trained model"""
    result = model.generate_content(test_input)
    return result.text

if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env file")
    
    model = train_model(api_key)
    test_result = test_model(model, "מה שמך?")
    print(f"Model response: {test_result}") 