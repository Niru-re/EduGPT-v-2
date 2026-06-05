from inference.chat import load_inference_model, generate_response
import json
import os

def evaluate():
    model, tokenizer = load_inference_model()
    
    test_prompts = [
        "Explain the process of photosynthesis.",
        "Summarize the main causes of the French Revolution.",
        "Create a quiz question about Python lists.",
        "Create a flashcard for Quantum Mechanics.",
        "What are the benefits of learning Linear Algebra?"
    ]
    
    results = []
    print("\nRunning evaluation...")
    
    for prompt in test_prompts:
        print(f"Testing: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        results.append({
            "instruction": prompt,
            "response": response
        })
    
    # Save results
    os.makedirs("report", exist_ok=True)
    with open("report/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nEvaluation complete. Results saved to report/evaluation_results.json")

if __name__ == "__main__":
    evaluate()
