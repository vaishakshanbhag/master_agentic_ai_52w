import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

#helper funtion to embed text using OpenAI embeddings
def ask(prompt, show=True):
    response= client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role":"user","content":prompt}]
    )
    answer= response.choices[0].message.content
    if show: print(answer)
    return answer

#baseline vs CoT (Chain of Thought) prompting

# q = "A farmer has 17 sheep. All but 9 run away. How many are left?"
# print("Baseline Prompting:")
# ask(q)

# print("\nChain of Thought Prompting:")
# ask(q + "\n\nReason step by step before answering.")

# q = """Alice put her laptop in her backpack. 
# Then she went to school. 
# Where is the laptop most likely now?"""
 
# print("\n--- Baseline ---")
# ask(q)
 
# print("\n--- With CoT ---")
# ask(q + "\n\nExplain your reasoning step by step.")


# prompt = """
# Solve: A train leaves at 3pm going 60 mph. 
# Another leaves at 4pm going 90 mph. 
# When will the faster train catch up?
 
# Answer format:
# Reasoning:
# Final Answer:
# """
# print(ask(prompt))


tasks = [
    "What is 23*17?",
    "If it rains, the ground gets wet. It is raining. What can we conclude?",
    "A red ball is in the box. The box is tipped over. Where is the ball?"
]
 
for t in tasks:
    print("\nQ:", t)
    baseline = ask(t, show=False)
    cot = ask(t + "\nThink step by step.", show=False)
    print("Baseline:", baseline.strip())
    print("CoT:", cot.strip())