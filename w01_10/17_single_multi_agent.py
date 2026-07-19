"""
Lab 1: Simulating Single vs Multi-Agent Interactions
Goal: Experience how multiple lightweight agents coordinate (or clash) versus a single agent tackling the same task.
You will build a tiny turn-based simulator with a blackboard (shared memory), roles, and a message protocol.
"""

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a local .env file if present.
load_dotenv()
llm = OpenAI()


def chat(system: str, user: str, temp: float = 0):
    """Send a prompt to the LLM and return the assistant's reply text."""
    # Create a single chat completion request with the system and user messages.
    r = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
    )
    return r.choices[0].message.content.strip()


@dataclass
class Message:
    """Represents one message exchanged between agents or the user."""

    sender: str
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class Blackboard:
    """Shared memory container for a task run."""

    data: Dict[str, Any] = field(default_factory=dict)
    log: List[Message] = field(default_factory=list)

    def write(self, msg: Message):
        """Append a message to the blackboard's conversation log."""
        self.log.append(msg)

    def update(self, **kwargs):
        """Merge new key/value data into the shared blackboard state."""
        self.data.update(kwargs)


# We’ll use a Researcher, Planner, and Critic. The single-agent will combine all roles.
ROLE_RESEARCHER = (
    "Role: Researcher. Read the task & propose 3 concise facts or sources "
    "to inform a solution. Output as bullet points. No final answer."
)

ROLE_PLANNER = (
    "Role: Planner. Read the latest facts and propose an ordered plan with 3 to 5 steps. "
    "State assumptions and risks briefly."
)

ROLE_CRITIC = (
    "Role: Critic. Review the plan for gaps, conflicts, or missing data. "
    "Return 2 to 3 actionable improvements. No final answer."
)

ROLE_SINGLE = (
    "You are a single agent acting as Researcher, Planner, and Critic at once. "
    "Given the task, produce facts, a plan, then self-critique with improvements."
)


class BaseAgent:
    """Base class for a simple role-based agent."""

    def __init__(self, name: str, role: str):
        """Initialize the agent with a display name and a role prompt."""
        self.name, self.role = name, role

    def act(self, task: str, bb: Blackboard) -> Message:
        """Generate a response for the given task using the shared blackboard context."""
        # Build the prompt context from the task, current data, and recent messages.
        context = (
            f"Task:\n{task}\n\nBlackboard:\n{bb.data}\n\n"
            f"Conversation so far:\n"
            + "\n".join([f"- {m.sender}: {m.content[:200]}" for m in bb.log[-6:]])
        )
        out = chat(system=self.role, user=context)
        return Message(sender=self.name, content=out)


researcher = BaseAgent("Researcher", ROLE_RESEARCHER)
planner = BaseAgent("Planner", ROLE_PLANNER)
critic = BaseAgent("Critic", ROLE_CRITIC)
single = BaseAgent("Solo", ROLE_SINGLE)


def summarize_last(msgs: List[Message], role: str) -> str:
    """Create a compact summary of the most recent messages for the next agent."""
    if not msgs:
        return ""
    # Use the last two messages so the next agent sees the most relevant context.
    joined = "\n".join([m.content for m in msgs[-2:]])
    return chat(system=f"Summarize for {role}.", user=joined)


def multi_agent_run(task: str, rounds: int = 1) -> Blackboard:
    """Run a simple multi-agent workflow where researcher, planner, and critic take turns."""
    bb = Blackboard(data={"task": task, "facts": [], "plan": "", "critique": ""})

    for r in range(rounds):
        # Researcher collects initial facts and writes them to the blackboard.
        m1 = researcher.act(task, bb)
        bb.write(m1)
        bb.update(facts=bb.data.get("facts", []) + [m1.content])

        # Planner proposes an action plan using the latest context summary.
        ctx = summarize_last(bb.log, "Planner")
        m2 = planner.act(task + "\n\nContext:\n" + ctx, bb)
        bb.write(m2)
        bb.update(plan=m2.content)

        # Critic reviews the plan and suggests improvements.
        ctx = summarize_last(bb.log, "Critic")
        m3 = critic.act(task + "\n\nContext:\n" + ctx, bb)
        bb.write(m3)
        bb.update(critique=m3.content)

        # Planner revises its plan based on the critique.
        ctx = summarize_last(bb.log, "Planner")
        m4 = planner.act(task + "\n\nRevise plan per critique:\n" + ctx, bb)
        bb.write(m4)
        bb.update(plan=m4.content, round=r + 1)

    return bb


def single_agent_run(task: str) -> Blackboard:
    """Run a single-agent version that handles research, planning, and critique in one pass."""
    bb = Blackboard(data={"task": task})
    # Let the solo agent produce the final answer directly from the task.
    m = single.act(task, bb)
    bb.write(m)
    bb.update(final=m.content)
    return bb


# TASK = ("Design a 3-email outreach sequence for a B2B AI tool targeting "
#         "healthcare analytics leaders. Include subject lines and a clear CTA each.")
 
TASK = ("Create a 5-step incident response checklist for an LLM outage in a production environment. ")
bb_multi  = multi_agent_run(TASK, rounds=1)
bb_single = single_agent_run(TASK)
 
print("\n=== MULTI-AGENT (final plan) ===\n", bb_multi.data.get("plan",""))
print("\n=== MULTI-AGENT (critique) ===\n", bb_multi.data.get("critique",""))
print("\n=== SINGLE-AGENT ===\n", bb_single.data.get("final",""))


comparison_prompt = f"""
Compare two outputs for the same task.
 
[Multi-agent Plan]
{bb_multi.data.get('plan','')}
 
[Single-agent Output]
{bb_single.data.get('final','')}
 
Score each (1–5) on Coverage, Quality, Diversity, and provide a 3-sentence verdict.
"""
print("\n=== AUTO EVAL ===\n", chat("You are a fair evaluator.", comparison_prompt))