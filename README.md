# Model Fine-tuning and Evaluation Project (CS263)

## Overview
This project implements a fine-tuning pipeline for language models, with specific focus on Llama models and ChatGPT integration. It includes modular components for training, evaluation, and inference.

## Project Structure
```
├── _pycache__/           # Python cache directory
├── archive/             # Contains modularized components and logs
├── .DS_Store           
├── .env                 # Environment variables (Llama configuration)
├── .gitignore          
├── README.md           
├── chatgpt_communicate.py        # ChatGPT interaction module
├── environment.yml              # Conda environment specification
├── functions.py                # Training utility functions
├── model_configs.py            # Model configuration and checkpointing
├── translated_test_dataset.json # Test dataset for ChatGPT
└── unified_model_trainer.py     # Main training pipeline
```

## Prerequisites
- Python 3.8 or higher
- Git
- Hugging Face account and API token
- OpenAI API key (for ChatGPT integration)
- All provided API tokens are temporary (you are expected to create your own)

## Installation

### 1. Environment Setup
Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate [environment-name]
```

Alternatively, using pip:
```bash
pip install numpy pandas torch tqdm datasets huggingface_hub transformers evaluate python-dotenv
```

### 2. Configuration
Create a `.env` file with your credentials:
```plaintext
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_openai_key_here
# Add any additional Llama-specific configurations
```

## Data and Model Setup

### 1. Dataset
- The test dataset is provided in `translated_test_dataset.json`
- Training data is automatically fetched using your Hugging Face token

### 2. Model Configuration
- Model configurations are defined in `model_configs.py`
- Checkpoints are managed through this module
- Configure your specific model parameters before training

## Running the Project

### 1. Training Pipeline
```bash
python unified_model_trainer.py
```
This script:
- Utilizes functions from `functions.py` for training utilities
- Applies configurations from `model_configs.py`
- Runs the complete training and evaluation pipeline

### 2. ChatGPT Integration
```bash
python chatgpt_communicate.py
```
- Handles communication with OpenAI's GPT models
- Uses the test dataset from `translated_test_dataset.json`

## Model Training and Evaluation

### Training Configuration
The training process is configured in `unified_model_trainer.py` and uses utility functions from `functions.py`. Key components include:
- Model initialization and configuration
- Training loop implementation
- Evaluation metrics calculation

### Checkpointing
Checkpoints are managed through `model_configs.py`, which handles:
- Model state saving
- Configuration persistence
- Checkpoint recovery

## Troubleshooting

Common issues and solutions:
1. **Environment Setup**: Ensure all dependencies in `environment.yml` are properly installed
2. **API Authentication**: Verify your API keys in `.env` are correct and have necessary permissions
3. **Data Loading**: Check that your Hugging Face token has access to required datasets
