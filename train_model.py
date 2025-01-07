import google.generativeai as genai
import time
import pandas as pd
import seaborn as sns
from typing import List, Dict
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# טעינת משתני הסביבה מקובץ .env
load_dotenv()

def create_training_examples() -> List[Dict[str, str]]:
    """יוצר דוגמאות אימון"""
    return [
        {"text_input": "מה שמך?", "output": "שמי הוא עוזר AI"},
        {"text_input": "מה אתה יכול לעשות?", "output": "אני יכול לעזור לך במגוון משימות כמו כתיבה, תכנות, וניתוח מידע"},
        {"text_input": "איך אתה יכול לעזור לי?", "output": "אני יכול לסייע בכתיבת קוד, פתרון בעיות תכנות, ניתוח נתונים, ומענה על שאלות"},
        {"text_input": "באיזו שפה אתה מדבר?", "output": "אני מדבר עברית ויכול לתקשר בשפה זו באופן שוטף"},
        {"text_input": "מה השעה?", "output": "אני לא יכול לדעת מה השעה הנוכחית כי אין לי גישה לשעון בזמן אמת"},
        {"text_input": "תספר לי בדיחה", "output": "למה התוכניתן הלך לרופא? כי היה לו באג..."},
    ]

def train_model(
    api_key: str,
    display_name: str = "my-assistant",
    epoch_count: int = 20,
    batch_size: int = 2,
    learning_rate: float = 0.001
) -> genai.GenerativeModel:
    """מאמן מודל חדש"""
    
    genai.configure(api_key=api_key)
    training_data = create_training_examples()
    
    print("Starting model training...")
    operation = genai.create_tuned_model(
        display_name=display_name,
        source_model="models/gemini-1.5-flash-001-tuning",
        epoch_count=epoch_count,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_data=training_data
    )

    try:
        # Simplified progress tracking
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
    """בודק את המודל המאומן"""
    result = model.generate_content(test_input)
    return result.text

if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env file")
    
    model = train_model(api_key)
    
    # בדיקת המודל
    test_result = test_model(model, "מה שמך?")
    print(f"תשובת המודל: {test_result}") 