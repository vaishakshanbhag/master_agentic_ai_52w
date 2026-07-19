"""Lab 3: Observing Emergent Behavior in MAS
Goal: Simulate multiple agents with minimal coordination and observe how their interactions generate patterns not explicitly coded."""

import time, random, os
from dataclasses import dataclass, field
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
 

llm = OpenAI()
 
@dataclass
class Agent:
    name: str
    persona: str
    memory: List[str] = field(default_factory=list)
 
    def speak(self, context:str)->str:
        prompt = (
            f"You are {self.name}, {self.persona}.\n"
            f"Conversation so far:\n{context}\n\n"
            "Respond concisely (2 to 4 sentences)."
        )
        r = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        msg = r.choices[0].message.content.strip()
        self.memory.append(msg)
        return msg
 
agents = [
    Agent("Optimist", "always enthusiastic, highlights positives, proposes bold ideas"),
    Agent("Skeptic", "critical thinker, points out flaws, asks tough questions"),
    Agent("Mediator", "balances views, seeks compromise and common ground")
]

def run_conversation(topic:str, turns:int=6):
    transcript = []
    for t in range(turns):
        speaker = agents[t % len(agents)]
        context = "\n".join([f"{a}: {m}" for a, m in transcript[-6:]])
        msg = speaker.speak(f"Topic: {topic}\n\n{context}")
        transcript.append((speaker.name, msg))
        # print(f"{speaker.name}: {msg}\n")
    return transcript
 
topic = "Should we use multi-agent AI for healthcare decision support?"
transcript = run_conversation(topic, turns=9)


analysis_prompt = (
f"Here is a multi-agent conversation transcript:\n\n" +
"\n".join([f"{n}: {m}" for n,m in transcript]) +
"\n\nAnalyze:\n- Instances of collaboration or synergy\n"
"- Instances of conflict or contradiction\n"
"- Any surprising emergent patterns\n"
"- Who influenced the group most\n"
)
r = llm.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":analysis_prompt}]
)
print("\n=== Emergent Behavior Analysis ===\n", r.choices[0].message.content)

