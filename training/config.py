import torch
import os

class Config:
    # Model Configuration
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    # LoRA Configuration
    LORA_R = 16 # Increased for better capacity
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Training Configuration
    OUTPUT_DIR = "./models/edugpt-lora"
    BATCH_SIZE = 1 # Keep small for 4GB VRAM
    GRADIENT_ACCUMULATION_STEPS = 8 # Increase to compensate for small batch size
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3
    MAX_SEQ_LENGTH = 1024 # Increased for longer summaries/explanations
    FP16 = True
    BF16 = False # SmolLM2 works well with FP16
    LOGGING_STEPS = 5
    SAVE_STEPS = 50
    EVAL_STEPS = 50
    
    # Optimization
    GRADIENT_CHECKPOINTING = True
    OPTIMIZER = "paged_adamw_8bit" # Very efficient for low VRAM
    
    # Dataset Configuration
    DATA_PATH = "data/education_dataset.json"
    TEST_SIZE = 0.05
    
    # Quantization Configuration (QLoRA)
    USE_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = torch.float16
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_USE_DOUBLE_QUANT = True

    @staticmethod
    def get_device_map():
        if torch.cuda.is_available():
            # Check VRAM
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Detected GPU with {vram:.2f} GB VRAM")
            return "auto"
        return {"": "cpu"}
