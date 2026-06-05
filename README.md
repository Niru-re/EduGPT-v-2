# EduGPT – Fine-Tuned Educational Assistant 🎓

EduGPT is a production-ready, end-to-end educational AI assistant built by fine-tuning the **SmolLM2-135M** model. It is specifically designed to run on consumer-grade hardware (like a 4GB NVIDIA GPU) using **QLoRA** and **4-bit quantization**.

## 🚀 Features
- **Topic Explanation:** Breaks down complex subjects into simple, high-school-level terms.
- **Notes Summarization:** Condenses educational text into key principles and insights.
- **Quiz Generation:** Creates multiple-choice questions with explanations for active learning.
- **Flashcard Generation:** Automatically generates study flashcards for better retention.
- **Educational QA:** High-accuracy answering for academic questions.
- **Modern UI:** Premium Glassmorphism dark theme with responsive layout and real-time chat.
- **RAG Ready:** Architecture prepared for PDF ingestion and FAISS semantic search.

## 🛠️ Tech Stack
- **AI Core:** SmolLM2-135M-Instruct (Hugging Face)
- **Fine-Tuning:** LoRA, QLoRA (PEFT, BitsAndBytes)
- **Training Framework:** TRL SFTTrainer, PyTorch, Accelerate
- **Interface:** Streamlit (Custom Glassmorphism CSS)
- **Data:** Synthetic Instruction-Tuning Dataset (1200+ samples)

## 📁 Project Architecture
```text
EduGPT/
├── app/                # Streamlit UI & Glassmorphism design
├── data/               # Dataset generator & educational samples
├── inference/          # Chat scripts & evaluation metrics
├── models/             # Saved LoRA adapters (fine-tuned weights)
├── report/             # Evaluation & training reports
├── training/           # QLoRA training scripts & configuration
├── requirements.txt    # Production dependencies
└── README.md           # Documentation
```

## ⚙️ Installation & Setup

1. **Clone & Navigate:**
   ```bash
   git clone <repo-url>
   cd EduGPT
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Training Data:**
   ```bash
   python data/generate_dataset.py
   ```

4. **Start Fine-Tuning:**
   ```bash
   python training/train.py
   ```
   *Note: Automatically detects GPU/VRAM. For 4GB cards, it uses gradient checkpointing and 8-bit optimization.*

5. **Launch EduGPT Assistant:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 📊 Hardware Optimization
- **Memory:** Uses `paged_adamw_8bit` optimizer and `gradient_checkpointing` to minimize VRAM usage.
- **Quantization:** Employs `NF4` 4-bit quantization (QLoRA) for efficient weights loading.
- **Batching:** Uses small per-device batch sizes with high gradient accumulation (8 steps) to maintain effective batch size without crashing.

## 📄 Resume Content (AI Engineer)
- **Project: EduGPT - Fine-Tuned Educational Assistant**
- Built an end-to-end LLM pipeline using **SmolLM2-135M** and **QLoRA** for educational domain adaptation.
- Optimized model training for **4GB VRAM** constraints using **PEFT**, **bitsandbytes**, and **gradient checkpointing**.
- Developed a synthetic dataset generator producing **1200+ instruction-tuning pairs** across Science, History, Tech, and Math.
- Designed a **Streamlit** dashboard with a custom **Glassmorphism UI**, featuring multi-tab educational tools (Summarizer, Quiz Gen).
- Implemented robust error handling for CUDA memory management and fallback mechanisms for low-resource environments.

## 🔮 Future Roadmap
- **Full RAG Integration:** Use FAISS to index uploaded PDFs for document-grounded QA.
- **DPO Alignment:** Further align the model using Direct Preference Optimization.
- **Vision Support:** Integrate image-to-text for analyzing study diagrams.

---
*Developed by a Senior AI & Full-Stack Engineer for the next generation of digital learning.*
