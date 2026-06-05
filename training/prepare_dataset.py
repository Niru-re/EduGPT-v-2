import json
import os
import sys

# Add project root to path BEFORE other local imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasets import Dataset
from training.config import Config

def load_and_prepare_dataset():
    if not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {Config.DATA_PATH}. Run generate_dataset.py first.")
        
    # Load the synthetic dataset
    with open(Config.DATA_PATH, "r") as f:
        data = json.load(f)
    
    # Format for instruction tuning (ChatML style for SmolLM2)
    formatted_data = []
    for item in data:
        # SmolLM2 often expects a chat-like structure or specific delimiters
        # Using a simple instruction-response format that SFTTrainer can handle
        text = f"<|user|>\n{item['instruction']}<|endoftext|>\n<|assistant|>\n{item['response']}<|endoftext|>"
        formatted_data.append({"text": text})
    
    # Create HF Dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Shuffle and Split
    dataset = dataset.shuffle(seed=42)
    split_dataset = dataset.train_test_split(test_size=Config.TEST_SIZE)
    
    return split_dataset

if __name__ == "__main__":
    dataset = load_and_prepare_dataset()
    print(f"Dataset prepared: {dataset}")
    print(f"Example:\n{dataset['train'][0]['text']}")
