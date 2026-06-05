import torch
import os
import sys
import numpy as np
import torch.serialization

# Allowlist numpy types for safe loading globally
torch.serialization.add_safe_globals([
    np._core.multiarray._reconstruct, 
    np.ndarray, 
    np.dtype,
    np.dtypes.UInt32DType if hasattr(np.dtypes, 'UInt32DType') else None,
    np.dtypes.Int64DType if hasattr(np.dtypes, 'Int64DType') else None
])

# Add project root to path BEFORE other local imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from training.config import Config
from training.prepare_dataset import load_and_prepare_dataset

def train(smoke_test=False):
    set_seed(42)
    
    # 1. Load Tokenizer
    print(f"Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. BitsAndBytes Configuration (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=Config.USE_4BIT,
        bnb_4bit_compute_dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_quant_type=Config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=Config.BNB_4BIT_USE_DOUBLE_QUANT,
    )

    # 3. Load Base Model
    device_map = Config.get_device_map()
    print(f"Loading model: {Config.MODEL_NAME} on {device_map}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map=device_map,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Retrying without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            device_map=device_map,
            trust_remote_code=True
        )
    
    # 4. Prepare model for kbit training
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # 5. LoRA Configuration
    peft_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. Load Dataset
    print("Preparing dataset...")
    dataset = load_and_prepare_dataset()
    
    if smoke_test:
        print("Smoke test enabled: using 10 samples only.")
        dataset["train"] = dataset["train"].select(range(min(10, len(dataset["train"]))))
        dataset["test"] = dataset["test"].select(range(min(5, len(dataset["test"]))))
        num_epochs = 1
        max_steps = 5
    else:
        num_epochs = Config.NUM_EPOCHS
        max_steps = -1

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        fp16=Config.FP16 and torch.cuda.is_available(),
        logging_steps=Config.LOGGING_STEPS,
        save_steps=Config.SAVE_STEPS,
        evaluation_strategy="steps" if not smoke_test else "no",
        eval_steps=Config.EVAL_STEPS,
        optim=Config.OPTIMIZER if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=Config.GRADIENT_CHECKPOINTING,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        report_to="none",
        push_to_hub=False,
    )

    # 8. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=Config.MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 9. Train
    print("Starting training...")
    try:
        # Check if checkpoint exists to resume
        checkpoint_dir = os.path.join(Config.OUTPUT_DIR, "checkpoint-350")
        resume_from = checkpoint_dir if os.path.exists(checkpoint_dir) else None
        
        if resume_from:
            print(f"Resuming training from: {resume_from}")
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            print("No checkpoint found at 350. Starting fresh training.")
            trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA Out of Memory! Try reducing MAX_SEQ_LENGTH or BATCH_SIZE.")
            torch.cuda.empty_cache()
        raise e

    # 10. Save the adapter
    if not smoke_test:
        print(f"Saving fine-tuned adapter to {Config.OUTPUT_DIR}")
        trainer.model.save_pretrained(Config.OUTPUT_DIR)
        tokenizer.save_pretrained(Config.OUTPUT_DIR)
    else:
        print("Smoke test finished successfully!")

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    smoke = "--smoke" in sys.argv
    train(smoke_test=smoke)
