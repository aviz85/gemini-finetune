# Hebrew AI Assistant Fine-tuning

Fine-tune a Gemini model to create a Hebrew-speaking AI assistant.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your Google API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```

## Configuration

### Training Parameters
Create `config.json` to customize training:
```json
{
    "training_params": {
        "display_name": "my-assistant",
        "epoch_count": 20,
        "batch_size": 2,
        "learning_rate": 0.001,
        "source_model": "models/gemini-1.5-flash-001-tuning"
    }
}
```

### Dataset
Create `dataset.json` for training examples:
```json
{
    "training_examples": [
        {"text_input": "שאלה בעברית", "output": "תשובה בעברית"}
    ]
}
```

## Usage

1. Train the model:
```bash
python train_model.py
```
This will:
- Use config from `config.json` (or defaults if missing)
- Train using `dataset.json` (or defaults if missing)
- Save training loss plot to `training_loss.png`
- Output your model name (save it for chat)

2. Chat with the model:
```bash
# Replace with your model name from training output
python use_model.py --model "tunedModels/myassistant-xxx"
```

Chat commands:
- Type your message and press Enter
- Press Enter without text to skip
- Type 'quit' to exit

3. List available models:
```bash
python list_models.py
```

## Notes

- Model names are prefixed with `tunedModels/`
- Training loss plots saved as `training_loss.png`
- Default values used if config files missing
