import torch
import os
import sys

# Add project root to path BEFORE other local imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from training.config import Config

def load_inference_model():
    # Load base model
    print(f"Loading base model: {Config.MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Load LoRA adapter if it exists, otherwise use base model
    if os.path.exists(Config.OUTPUT_DIR):
        print(f"Loading LoRA adapter from: {Config.OUTPUT_DIR}")
        model = PeftModel.from_pretrained(base_model, Config.OUTPUT_DIR)
    else:
        print("LoRA adapter not found. Using base model for inference.")
        model = base_model
        
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_new_tokens=512):
    # Use the same format as training
    prompt = f"<|user|>\n{instruction}<|endoftext|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract only the response part after <|assistant|>
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    else:
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Cleanup
    response = response.replace("<|endoftext|>", "").strip()
    return response

if __name__ == "__main__":
    model, tokenizer = load_inference_model()
    print("\n--- EduGPT Chat Interface ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = generate_response(model, tokenizer, user_input)
        print(f"EduGPT: {response}\n")
