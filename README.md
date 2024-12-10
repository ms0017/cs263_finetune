Overview
This project utilizes several Python libraries and Hugging Face tools to create and fine-tune language models for various NLP Finetuning alongside of evaluation. Below are the installation instructions and a list of the required packages.

pip install numpy pandas torch tqdm datasets huggingface_hub transformers evaluate python-dotenv

once all packges are install, proceed by running

python3 unified_model_trainer.py which will run the training and evaluation loops.

To access fine-tuned GPT model, first acquire GPT access token,

then replace your token with client = OpenAI(api_key = "") in chatgpt_communicate.py



