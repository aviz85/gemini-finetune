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

## Usage

1. Train the model:
```bash
python train_model.py
```
This will:
- Train the model using examples from `dataset.json`
- Save training loss plot to `training_loss.png`
- Print the model name when done

2. Chat with the model:
```bash
python use_model.py
```

3. List available models:
```bash
python list_models.py
```

## Customizing Training

- Edit `dataset.json` to add/modify training examples
- Adjust training parameters in `train_model.py`:
  - `epoch_count`: Number of training epochs
  - `batch_size`: Batch size for training
  - `learning_rate`: Learning rate for training

## Notes

- The model name will be prefixed with `tunedModels/` in the output
- The training loss plot will be saved as `training_loss.png`
- The model will be saved in the `tunedModels` directory
