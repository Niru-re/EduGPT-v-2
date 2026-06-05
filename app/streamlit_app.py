import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys
import json
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.config import Config

# --- Page Config ---
st.set_page_config(
    page_title="EduGPT Premium | SaaS Learning Workspace",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = '💬 Chat Workspace'

# --- CSS: Premium SaaS UI ---
def apply_saas_ui():
    theme = st.session_state.theme
    bg_color = "#0a0a0a" if theme == 'dark' else "#f8fafc"
    card_bg = "rgba(255, 255, 255, 0.08)" if theme == 'dark' else "rgba(255, 255, 255, 0.7)"
    text_color = "#ffffff" if theme == 'dark' else "#0f172a"
    border_color = "rgba(255, 255, 255, 0.12)" if theme == 'dark' else "rgba(0, 0, 0, 0.05)"
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        * {{
            font-family: 'Outfit', sans-serif !important;
        }}

        .stApp {{
            background: {bg_color};
            color: {text_color};
            transition: all 0.5s ease;
        }}

        /* Animated Blobs */
        .blob {{
            position: fixed;
            width: 400px;
            height: 400px;
            border-radius: 50%;
            filter: blur(80px);
            z-index: -1;
            opacity: {0.3 if theme == 'dark' else 0.15};
            animation: float 20s infinite alternate;
        }}
        .blob-purple {{ background: #f59e0b; top: -100px; left: -100px; }}
        .blob-blue {{ background: #fbbf24; bottom: -100px; right: -100px; animation-delay: -5s; }}
        .blob-pink {{ background: #d97706; top: 40%; left: 60%; animation-delay: -10s; }}

        @keyframes float {{
            from {{ transform: translate(0, 0) scale(1); }}
            to {{ transform: translate(100px, 100px) scale(1.2); }}
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: {card_bg} !important;
            backdrop-filter: blur(25px) !important;
            -webkit-backdrop-filter: blur(25px) !important;
            border-right: 1px solid {border_color} !important;
            width: 300px !important;
        }}

        /* Premium Glass Cards */
        .glass-card {{
            background: {card_bg};
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border: 1px solid {border_color};
            border-radius: 24px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-5px);
        }}

        /* Chat Layout (ChatGPT Style) */
        .chat-container {{
            max-width: 1000px;
            margin: 0 auto;
            padding-bottom: 150px;
        }}

        .chat-bubble-user {{
            background: { "rgba(245, 158, 11, 0.2)" if theme == 'dark' else "rgba(245, 158, 11, 0.1)" } !important;
            border: 1px solid rgba(245, 158, 11, 0.2) !important;
            border-radius: 20px 20px 0 20px !important;
            padding: 16px 24px !important;
            margin-bottom: 1.5rem !important;
            align-self: flex-end;
            color: {text_color} !important;
        }}

        .chat-bubble-assistant {{
            background: transparent !important;
            border: none !important;
            padding: 16px 24px !important;
            margin-bottom: 1.5rem !important;
            color: {text_color} !important;
            line-height: 1.6;
        }}

        /* Floating Input (Fixed Selector) */
        [data-testid="stChatInput"] {{
            position: fixed !important;
            bottom: 25px !important;
            width: 800px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            z-index: 999 !important;
            background: { "rgba(255,255,255,0.05)" if theme == 'dark' else "rgba(255,255,255,0.9)" } !important;
            backdrop-filter: blur(24px) !important;
            border: 1px solid {border_color} !important;
            border-radius: 24px !important;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important;
        }}

        /* Stats & Widgets */
        .stat-badge {{
            padding: 8px 16px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            border: 1px solid {border_color};
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}

        /* Typing Animation */
        .typing-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #f59e0b;
            border-radius: 50%;
            margin-right: 4px;
            animation: blink 1.4s infinite both;
        }}
        .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
        .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes blink {{
            0%, 80%, 100% {{ opacity: 0; }}
            40% {{ opacity: 1; }}
        }}

        /* Show Sidebar Toggle but hide other header fluff */
        header[data-testid="stHeader"] {{
            background: transparent !important;
            color: {text_color} !important;
        }}
        footer {{ visibility: hidden; }}
        .stDeployButton {{ display: none; }}
    </style>
    
    <div class="blob blob-purple"></div>
    <div class="blob blob-blue"></div>
    <div class="blob blob-pink"></div>
    """, unsafe_allow_html=True)

# --- Engine Logic ---
@st.cache_resource
def load_engine():
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
        
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME,
                device_map=device_map,
                trust_remote_code=True
            )
        
        if os.path.exists(Config.OUTPUT_DIR):
            st.sidebar.success("✅ EduGPT LoRA Model Loaded!")
            model = PeftModel.from_pretrained(model, Config.OUTPUT_DIR)
        else:
            st.sidebar.warning("⚠️ Using Base Model (Training Pending)")
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Engine Load Error: {e}")
        return None, None

def ai_generate(model, tokenizer, prompt, temp=0.3, max_tokens=512):
    # Use standard ChatML format which SmolLM2-Instruct is pre-trained on
    # This helps even if the fine-tuning was slightly different
    system_prompt = "You are EduGPT, a helpful and professional educational AI assistant. Provide clear, accurate, and concise explanations."
    
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Stricter parameters for 135M model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    res = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract response after assistant token
    if "<|im_start|>assistant" in res:
        response = res.split("<|im_start|>assistant")[-1].strip()
    elif "<|assistant|>" in res:
        response = res.split("<|assistant|>")[-1].strip()
    else:
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Final cleanup of all possible tags
    stop_tags = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|user|>", "<|assistant|>", "assistant", "user", "system"]
    for tag in stop_tags:
        response = response.replace(tag, "")
    
    # Remove any trailing nonsensical code-like hallucinations if they appear
    if "</tbody>" in response:
        response = response.split("</tbody>")[0].strip()
    if "let " in response and "new Date()" in response:
        response = response.split("let ")[0].strip()
        
    return response.strip()

# --- Helper Logic for Tools ---
def get_quiz(model, tokenizer, topic):
    prompt = f"Generate a simple educational quiz with 3 multiple choice questions about {topic}. For each question, provide 4 options (A, B, C, D) and specify the correct answer."
    return ai_generate(model, tokenizer, prompt, temp=0.2, max_tokens=600)

def get_flashcards(model, tokenizer, topic):
    prompt = f"Create 3 educational flashcards for: {topic}. Format each card clearly with 'Question:' and 'Answer:'."
    return ai_generate(model, tokenizer, prompt, temp=0.3, max_tokens=500)

def get_summary(model, tokenizer, text):
    prompt = f"Summarize this educational text into 3 clear bullet points:\n\n{text}"
    return ai_generate(model, tokenizer, prompt, temp=0.1, max_tokens=350)

# --- Application UI ---
def main():
    apply_saas_ui()
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"<h1 style='font-size: 24px; margin-bottom: 20px;'>EduGPT ✨</h1>", unsafe_allow_html=True)
        
        st.session_state.active_tab = st.radio(
            "Navigation",
            ["🏠 Dashboard", "💬 Chat Workspace", "📝 Summarizer", "🧠 Flashcards", "❓ Quiz Generator", "📄 PDF RAG", "📊 Analytics"],
            label_visibility="collapsed"
        )
        
        st.divider()
        if st.button("🌙 Toggle Dark/Light Mode", use_container_width=True):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
            
        st.divider()
        st.markdown("<h4 style='font-size: 14px; color: #8b5cf6;'>AI AGENT STATUS</h4>", unsafe_allow_html=True)
        st.checkbox("Browse Web", value=True)
        st.checkbox("Search PDFs", value=True)
        st.checkbox("Research Mode", value=False)
        
    # --- Main Workspace ---
    model, tokenizer = load_engine()

    if st.session_state.active_tab == "💬 Chat Workspace":
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        if not st.session_state.messages:
            st.markdown("""
                <div style='text-align: center; margin-top: 100px;'>
                    <h1 style='font-size: 42px; font-weight: 800;'>Welcome Back 👋</h1>
                    <p style='font-size: 18px; opacity: 0.6;'>What would you like to learn today?</p>
                </div>
            """, unsafe_allow_html=True)

        for m in st.session_state.messages:
            bubble_class = "chat-bubble-user" if m["role"] == "user" else "chat-bubble-assistant"
            st.markdown(f"""
                <div style="display: flex; flex-direction: column; {'align-items: flex-end' if m['role'] == 'user' else 'align-items: flex-start'}">
                    <div class="{bubble_class}">{m['content']}</div>
                </div>
            """, unsafe_allow_html=True)

        if prompt := st.chat_input("Ask EduGPT anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            thinking = st.empty()
            thinking.markdown("""
                <div style='margin-bottom:20px'>
                    <span class='typing-dot'></span>
                    <span class='typing-dot'></span>
                    <span class='typing-dot'></span>
                    Thinking...
                </div>
            """, unsafe_allow_html=True)
            
            if model:
                response = ai_generate(model, tokenizer, st.session_state.messages[-1]["content"])
            else:
                response = "Model loading..."
                
            thinking.empty()
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.active_tab == "🏠 Dashboard":
        # Analytics Cards from User Screenshot
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("""<div class='glass-card'><h4>Average Rating</h4><h1>4.8</h1>⭐⭐⭐⭐⭐</div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class='glass-card'><h4>Questions</h4><h1>128</h1></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class='glass-card'><h4>Flashcards</h4><h1>52</h1></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown("""<div class='glass-card'><h4>PDFs</h4><h1>17</h1></div>""", unsafe_allow_html=True)

        col_main, col_right = st.columns([2, 1])
        with col_main:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Weekly Learning Progress")
            st.markdown("""
                <div style="text-align: center; padding: 40px;">
                    <div style="position: relative; width: 150px; height: 150px; margin: 0 auto;">
                        <svg viewBox="0 0 36 36" style="width: 150px; height: 150px;"><path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="rgba(245, 158, 11, 0.1)" stroke-width="3" /><path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#f59e0b" stroke-width="3" stroke-dasharray="75, 100" /></svg>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 24px; font-weight: 800;">75%</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_right:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Learning Stats")
            st.markdown("""<div class="stat-badge"><span>Learning Score</span><b>92%</b></div><div class="stat-badge"><span>Today's Questions</span><b>14</b></div><div class="stat-badge"><span>Quiz Accuracy</span><b>89%</b></div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.active_tab == "📄 PDF RAG":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("PDF RAG Workspace")
        st.file_uploader("Upload study materials", type="pdf")
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.active_tab == "📝 Summarizer":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("AI Text Summarizer")
        text_to_sum = st.text_area("Paste your educational content here...", height=250)
        if st.button("✨ Summarize Now", use_container_width=True):
            if text_to_sum and model:
                with st.spinner("Analyzing and condensing..."):
                    summary = get_summary(model, tokenizer, text_to_sum)
                    st.markdown("### 📋 Key Summary Points")
                    st.write(summary)
            else:
                st.warning("Please provide text and ensure model is loaded.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.active_tab == "❓ Quiz Generator":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Quiz Generator")
        topic = st.text_input("Topic", placeholder="e.g. Quantum Physics, French Revolution")
        if st.button("✨ Generate Quiz", use_container_width=True):
            if topic and model:
                with st.spinner("Generating questions..."):
                    quiz = get_quiz(model, tokenizer, topic)
                    st.markdown("### 📝 Practice Quiz")
                    st.write(quiz)
            else:
                st.warning("Please enter a topic.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.active_tab == "🧠 Flashcards":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Flashcard Generator")
        topic = st.text_input("Topic for Flashcards", placeholder="e.g. Biology terms, History dates")
        if st.button("✨ Create Flashcards", use_container_width=True):
            if topic and model:
                with st.spinner("Creating flashcards..."):
                    cards = get_flashcards(model, tokenizer, topic)
                    st.markdown("### 🧠 Study Cards")
                    st.write(cards)
            else:
                st.warning("Please enter a topic.")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
