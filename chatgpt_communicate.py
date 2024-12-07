import gc
import logging
import os
import sys
import warnings
from typing import Dict

# Third-party library imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Hugging Face imports
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from huggingface_hub import login
from openai import OpenAI
import json

if __name__ == "__main__":
    login(token='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')
    dataset = load_dataset('IssakaAI/en-tw')
    test_size = 100
    
    
    test_dataset = dataset['test'].shuffle(seed=42).select(range(test_size))
    new_data = []
    client = OpenAI(api_key = "")
    for sample in test_dataset:
        english_sentence = sample['ENGLISH']
        ground_truth_twi = sample['TWI']
        completion = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::AVSlrQUw",
        messages =[
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": "translate the following sentence into TWI, only provide the translation and nothing else. Sentense: "+ english_sentence}
        ]
        )
        translated_twi = completion.choices[0].message.content
        new_data.append({
            "ENGLISH": english_sentence,
            "GROUND_TRUTH_TWI": ground_truth_twi,
            "TRANSLATED_TWI": translated_twi
        })
    with open("translated_test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


