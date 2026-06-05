import json
import random
import os
from tqdm import tqdm

# --- High Quality Knowledge Base ---
KNOWLEDGE_BASE = {
    "Science": {
        "Photosynthesis": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. It involves the conversion of carbon dioxide and water into glucose and oxygen.",
        "Cell Biology": "Cell biology is the study of cell structure and function, revolving around the concept that the cell is the fundamental unit of life. It covers organelles, cell cycle, and metabolic processes.",
        "Genetics": "Genetics is the branch of biology that studies genes, genetic variation, and heredity in organisms. It explains how traits are passed from parents to offspring through DNA.",
        "Evolution": "Evolution is the process by which different kinds of living organisms are thought to have developed and diversified from earlier forms during the history of the earth, primarily through natural selection.",
        "Thermodynamics": "Thermodynamics is the branch of physics that deals with heat, work, and temperature, and their relation to energy, radiation, and physical properties of matter.",
        "Newton's Laws": "Newton's laws of motion are three physical laws that, together, laid the foundation for classical mechanics. They describe the relationship between a body and the forces acting upon it."
    },
    "Mathematics": {
        "Algebra": "Algebra is a branch of mathematics dealing with symbols and the rules for manipulating those symbols. In its basic form, it involves solving equations and representing relationships with variables.",
        "Calculus": "Calculus is the mathematical study of continuous change, in the same way that geometry is the study of shape and algebra is the study of generalizations of arithmetic operations.",
        "Geometry": "Geometry is a branch of mathematics that studies the sizes, shapes, positions, angles, and dimensions of things. It includes points, lines, surfaces, and solids.",
        "Statistics": "Statistics is the science of collecting, analyzing, presenting, and interpreting data. It is crucial for making informed decisions based on numerical evidence.",
        "Probability": "Probability is the branch of mathematics concerning numerical descriptions of how likely an event is to occur, or how likely it is that a proposition is true.",
        "Trigonometry": "Trigonometry is a branch of mathematics that studies relationships between side lengths and angles of triangles, specifically right-angled triangles."
    },
    "Technology": {
        "Python": "Python is a high-level, interpreted, general-purpose programming language. Its design philosophy emphasizes code readability with its use of significant indentation.",
        "Machine Learning": "Machine Learning is a branch of Artificial Intelligence that enables computers to learn from data and improve performance without explicit programming, using algorithms to find patterns.",
        "Artificial Intelligence": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions, such as problem-solving and learning.",
        "Data Structures": "Data structures are specialized formats for organizing, processing, retrieving, and storing data. Common types include arrays, linked lists, stacks, and queues.",
        "Algorithms": "An algorithm is a finite sequence of rigorous instructions, typically used to solve a class of specific problems or to perform a computation.",
        "Cybersecurity": "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks, aimed at accessing, changing, or destroying sensitive information."
    },
    "History": {
        "French Revolution": "The French Revolution was a period of far-reaching social and political upheaval in France and its colonies beginning in 1789, which overthrew the monarchy and established a republic.",
        "World War II": "World War II was a global conflict that lasted from 1939 to 1945, involving the vast majority of the world's countries and two opposing military alliances: the Allies and the Axis.",
        "Roman Empire": "The Roman Empire was the post-Republican period of ancient Rome. As a polity, it included large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia.",
        "Cold War": "The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies, the Western Bloc and the Eastern Bloc, after World War II."
    },
    "Economics": {
        "Inflation": "Inflation is a general increase in prices and fall in the purchasing value of money. It is typically measured as an annual percentage increase.",
        "Supply and Demand": "Supply and demand is an economic model of price determination in a market. It postulates that, holding all else equal, the unit price for a particular good will vary until it settles at an equilibrium.",
        "GDP": "Gross Domestic Product (GDP) is a monetary measure of the market value of all the final goods and services produced in a specific time period by a country.",
        "Fiscal Policy": "Fiscal policy is the use of government spending and taxation to influence the economy, typically used by governments to promote strong and sustainable growth and reduce poverty."
    }
}

def generate_hq_dataset(num_samples=3000):
    dataset = []
    categories = list(KNOWLEDGE_BASE.keys())
    
    print(f"Generating {num_samples} high-quality samples...")
    
    for _ in tqdm(range(num_samples)):
        category = random.choice(categories)
        topic = random.choice(list(KNOWLEDGE_BASE[category].keys()))
        fact = KNOWLEDGE_BASE[category][topic]
        
        data_type = random.choice(["explanation", "summary", "flashcard", "quiz", "qa"])
        
        if data_type == "explanation":
            instruction = f"Explain the concept of {topic}."
            response = f"{topic} is defined as follows: {fact} For example, in {category}, understanding {topic} helps in mastering more complex subjects."
        elif data_type == "summary":
            instruction = f"Provide a brief summary of {topic}."
            response = f"Summary: {topic} is a critical part of {category}. Key takeaway: {fact}"
        elif data_type == "flashcard":
            instruction = f"Create a flashcard for {topic}."
            response = f"Front: {topic}\nBack: {fact}"
        elif data_type == "quiz":
            instruction = f"Create a quiz question on {topic}."
            response = f"Question: Which of the following best describes {topic}?\nA) A type of energy\nB) {fact[:50]}...\nC) A historical figure\nD) None of the above\nCorrect Answer: B"
        elif data_type == "qa":
            instruction = f"What is {topic}?"
            response = f"{topic} refers to {fact.lower() if fact[0].isupper() else fact}"
            
        dataset.append({
            "instruction": instruction,
            "response": response,
            "category": category,
            "topic": topic,
            "type": data_type
        })
        
    return dataset

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    data = generate_hq_dataset(3000)
    
    with open("data/education_dataset.json", "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully generated {len(data)} HQ samples in data/education_dataset.json")
