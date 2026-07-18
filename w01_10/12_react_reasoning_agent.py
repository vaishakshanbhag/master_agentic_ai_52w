#practice code for ReAct Reasoning Agents
#Goal: Implement the ReAct pattern where the LLM alternates between thought and action. You’ll build a mini agent that reasons, calls tools, and returns an answer.

import math
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# Evaluate a mathematical expression provided as a string and return the result.
def calculaltor(expression:str):
    try:
        return str(eval(expression))  
    except Exception as e:
        return f"Error: {e}"
    
knowledge_base = {
    "LangChain": "A framework for building LLM-powered agents.",
    "RAG": "Retrieval-Augmented Generation, which uses external knowledge to improve answers."
}

def lookup(term:str):
    return knowledge_base.get(term, f"{term} not found in knowledge base.")


def react_agent(query, max_turns=5):
    conversation=[
        {"role":"system",
            "content":"You are a ReAct agent. You must think step by step and take actions when needed. "
            "Format:\nThought: ...\nAction: tool[input]\nObservation: ...\nFinal Answer: ..."}
    ]
    conversation.append({"role":"user", "content":query})

    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=conversation,
            temperature=1
        )
        reply = resp.choices[0].message.content
        print(reply)   # show reasoning trace
 
        # If the model chooses a tool, parse the action, run the tool, and feed the result
        # back into the conversation so the agent can continue reasoning with an observation.
        if "Action:" in reply:
            if "calculate[" in reply:
                expr = reply.split("calculate[")[1].split("]")[0]
                obs = calculaltor(expr)
            elif "lookup[" in reply:
                term = reply.split("lookup[")[1].split("]")[0]
                obs = lookup(term.strip())
            else:
                obs = "Unknown tool"
 
            conversation.append({"role":"assistant","content":reply})
            conversation.append({"role":"user","content":f"Observation: {obs}"})
        else:
            break

# react_agent("What is (15*3) + (120/6)?")
# react_agent("Explain LangChain in one sentence.")
react_agent("If LangChain is a framework, add 10*5 to that info.")