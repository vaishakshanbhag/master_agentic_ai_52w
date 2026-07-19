"""Implementing Agent Communication Protocols
Goal: Design and enforce structured agent-to-agent protocols (schemas + states + routing)."""

from __future__ import annotations
from typing import Literal, List, Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError
from time import time
import uuid, json, os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
llm = OpenAI()

# --- Message schema (validated envelope) ---
class Msg(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    proto: Literal["REQREP","PUBSUB","NEGOTIATE"]
    type: str                       # e.g., REQUEST, REPLY, PROPOSE, ACCEPT...
    sender: str
    to: List[str] = Field(default_factory=list)
    topic: Optional[str] = None     # for PUBSUB
    corr_id: Optional[str] = None   # correlation for replies
    payload: Dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time)
    
# --- Simple in-memory bus with mailboxes ---
@dataclass
class Mailbox:
    """A simple mailbox that stores messages for a single agent or participant."""

    name: str
    inbox: List[Msg] = field(default_factory=list)


class Bus:
    """An in-memory message bus that routes messages between registered mailboxes."""

    def __init__(self):
        """Initialize an empty routing table for agent mailboxes."""
        self.boxes: Dict[str, Mailbox] = {}

    def register(self, *names):
        """Create mailbox entries for one or more agent names if they do not exist yet."""
        for n in names:
            if n not in self.boxes:
                self.boxes[n] = Mailbox(n)

    def send(self, msg: Msg):
        """Deliver a message to its intended recipients based on the protocol type."""
        if msg.proto == "PUBSUB":
            # broadcast to all subscribers of topic; here: all except sender
            for name, box in self.boxes.items():
                if name != msg.sender:
                    box.inbox.append(msg)
        else:
            for dest in msg.to:
                self.boxes[dest].inbox.append(msg)

    def recv(self, name: str, max_items: int = 10) -> List[Msg]:
        """Return up to max_items messages from a mailbox and leave the rest queued."""
        box = self.boxes[name]
        out, box.inbox = box.inbox[:max_items], box.inbox[max_items:]
        return out

bus = Bus()

def chat_as(role: str, system: str, user: str, temp=0):
    """Call the LLM with a role-specific system prompt and return the assistant reply.

    Args:
        role: The persona or behavior to assign to the model.
        system: Additional instructions that describe the task or context.
        user: The actual user prompt or task request.
        temp: Sampling temperature that controls response randomness.

    Returns:
        A cleaned string containing the model's response content.
    """
    r = llm.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temp,
        messages=[
            {"role": "system", "content": f"Role: {role}. {system}"},
            {"role": "user", "content": user},
        ],
    )
    return r.choices[0].message.content.strip()
 
ROLES = {
    "Requester": "You create clear task specs and validate replies.",
    "Worker": "You fulfill requests precisely and report status.",
    "Auditor": "You observe traffic and flag malformed or unsafe messages."
}

#Protocol 1 — Request/Reply with state machine
REQREP_TRANSITIONS = {
    "REQUEST": {"REPLY","ERROR"},
}
 
def is_valid_reqrep(prev: Optional[Msg], cur: Msg) -> bool:
    """Validate whether a follow-up message is a legal transition in the request/reply protocol.

    Args:
        prev: The previous message in the conversation, if any.
        cur: The current message being checked.

    Returns:
        True when the current message follows the allowed state transition rules; otherwise False.
    """
    if cur.type == "REQUEST":
        return True  # initiating message
    if prev is None:
        return False
    allowed = REQREP_TRANSITIONS.get(prev.type, set())
    if cur.type not in allowed:
        return False
    if cur.corr_id != prev.id:
        return False
    return True
 
def send_req(sender: str, to: list, task: str) -> Msg:
    """Create and send a request message to one or more recipients.

    Args:
        sender: The name of the agent sending the request.
        to: A list of recipient agent names.
        task: The task description or instruction to be carried in the payload.

    Returns:
        The validated message object that was sent on the bus.
    """
    m = Msg(proto="REQREP", type="REQUEST", sender=sender, to=to, payload={"task": task})
    bus.send(m)
    return m

def send_reply(sender: str, req: Msg, result: dict, is_error: bool = False) -> Msg:
    """Create and send a reply or error message for a prior request.

    Args:
        sender: The agent sending the response.
        req: The original request message that this reply answers.
        result: The payload containing the answer or error details.
        is_error: If True, the message is marked as an error response instead of a normal reply.

    Returns:
        The message object that was sent on the bus.

    Raises:
        ValueError: If the reply does not follow the allowed request/reply protocol transition.
    """
    t = "ERROR" if is_error else "REPLY"
    m = Msg(proto="REQREP", type=t, sender=sender, to=[req.sender], corr_id=req.id, payload=result)
    if not is_valid_reqrep(req, m):
        raise ValueError("Invalid transition")
    bus.send(m)
    return m

bus.register("Requester","Worker","Auditor")
# # Enable the following lines to see a simple request/reply interaction between agents.
# # Requester posts a task
# req = send_req("Requester", ["Worker"], "Summarize issues with Agentic AI in 40 words.")
 
# # Worker consumes and replies
# incoming = bus.recv("Worker")
# assert incoming and incoming[0].type == "REQUEST"
# task = incoming[0].payload["task"]
 
# answer = chat_as("Worker", ROLES["Worker"], f"Task: {task}")
# reply = send_reply("Worker", incoming[0], {"answer": answer})
 
# # Requester reads reply
# print(bus.recv("Requester")[0].payload["answer"])


# Protocol 2 — Pub/Sub (broadcast topics + filters)
def publish(sender: str, topic: str, event: str, data: dict) -> Msg:
    """Publish a structured event to a topic so other agents can consume it.

    This helper creates a publish/subscribe message with a topic name and a payload
    containing an event label and associated data. The message is then routed through
    the in-memory bus so any registered agent can read it from the appropriate mailbox.

    Args:
        sender: The name of the agent emitting the event.
        topic: The topic channel to which the event belongs.
        event: A short label describing the event type, such as "job_complete".
        data: A dictionary with additional information relevant to the event.

    Returns:
        The published message object that was sent on the bus.
    """
    m = Msg(proto="PUBSUB", type="PUBLISH", sender=sender, topic=topic, payload={"event": event, "data": data})
    # schema guard
    assert "event" in m.payload and "data" in m.payload
    bus.send(m)
    return m


def consume_topic(name: str, topic: str, max_items: int = 5) -> List[Msg]:
    """Retrieve the most recent messages for a specific topic from an agent's mailbox.

    The function pulls queued messages from the mailbox of the given agent, filters them
    to those matching the requested topic, and returns up to a configurable number of
    matching messages in order of arrival.

    Args:
        name: The mailbox owner whose inbox should be checked.
        topic: The topic name to filter messages by.
        max_items: The maximum number of matching messages to return.

    Returns:
        A list of topic-matching messages from the mailbox, capped by max_items.
    """
    msgs = [m for m in bus.recv(name, 50) if m.topic == topic]
    return msgs[:max_items]
 
# # enable the following lines to see a simple publish/subscribe interaction between agents.
# publish("Worker", "telemetry", "job_started", {"ok": True, "latency_ms": 2300})
# publish("Worker", "telemetry", "job_complete", {"ok": True, "latency_ms": 2300})
# for m in consume_topic("Auditor", "telemetry"):
#     print("AUDIT EVENT:", m.payload)

# Protocol 3 — Negotiate/Contract (PROPOSE → COUNTER → ACCEPT/REJECT → CONFIRM)
# This protocol defines a stateful negotiation flow for multi-agent systems where
# agents must agree on a shared task, contract, or resource before proceeding.
# A sender begins with a proposal, the receiver may respond with a counter-offer,
# and the exchange continues only if each new message follows the allowed transition
# sequence. In this design, the helper functions such as propose(), counter(),
# accept(), and confirm() act like a lightweight contract protocol, ensuring that
# each step is validated and tied to the previous one.
#
# Example: a Planner agent proposes a task split to a Worker agent. The Worker may
# counter with revised terms, then the Planner accepts, and finally a confirmation
# message is sent to complete the agreement. This creates a structured negotiation
# loop that is easier to debug and safer to automate than free-form chatting.

NEG_TRANSITIONS = {
    "PROPOSE": {"COUNTER","ACCEPT","REJECT"},
    "COUNTER": {"COUNTER","ACCEPT","REJECT"},
    "ACCEPT":  {"CONFIRM"},
}
 
def valid_neg(prev: Optional[Msg], cur: Msg) -> bool:
    """Validate whether a negotiation message follows the allowed state machine.

    This function checks that a new negotiation message is a legal follow-up to the
    previous message in the conversation. A proposal is always allowed as the first
    step, while later messages must match the transition map and share the same
    correlation ID as the prior message.

    Args:
        prev: The previous negotiation message, if any.
        cur: The current negotiation message being evaluated.

    Returns:
        True if the transition is valid, otherwise False.
    """
    if cur.type == "PROPOSE":
        return True
    if prev is None:
        return False
    allowed = NEG_TRANSITIONS.get(prev.type, set())
    if cur.type not in allowed:
        return False
    if cur.corr_id != prev.id:
        return False
    return True


def propose(sender: str, to: list, spec: dict) -> Msg:
    """Create and send an initial negotiation proposal to one or more agents.

    Args:
        sender: The agent issuing the proposal.
        to: A list of recipient agents that should receive the proposal.
        spec: A dictionary describing the proposed terms, conditions, or contract details.

    Returns:
        The negotiation message object that was sent over the bus.
    """
    m = Msg(proto="NEGOTIATE", type="PROPOSE", sender=sender, to=to, payload=spec)
    bus.send(m)
    return m


def counter(sender: str, prev: Msg, spec: dict) -> Msg:
    """Create and send a counter-offer in response to an earlier proposal.

    Args:
        sender: The agent responding with a counter-offer.
        prev: The prior negotiation message that this counter-offer refers to.
        spec: The updated proposal details or revised terms.

    Returns:
        The counter-offer message object that was sent over the bus.

    Raises:
        ValueError: If the counter-offer does not follow the allowed negotiation state transitions.
    """
    m = Msg(proto="NEGOTIATE", type="COUNTER", sender=sender, to=[prev.sender], corr_id=prev.id, payload=spec)
    if not valid_neg(prev, m):
        raise ValueError("Invalid transition")
    bus.send(m)
    return m


def accept(sender: str, prev: Msg) -> Msg:
    """Create and send an acceptance message for a prior negotiation proposal.

    Args:
        sender: The agent accepting the proposal.
        prev: The prior negotiation message being accepted.

    Returns:
        The acceptance message object that was sent over the bus.

    Raises:
        ValueError: If the acceptance does not follow the allowed negotiation state transitions.
    """
    m = Msg(proto="NEGOTIATE", type="ACCEPT", sender=sender, to=[prev.sender], corr_id=prev.id, payload={"accepted": True})
    if not valid_neg(prev, m):
        raise ValueError("Invalid transition")
    bus.send(m)
    return m


def confirm(sender: str, prev: Msg) -> Msg:
    """Create and send a confirmation message after an accepted negotiation.

    This function finalizes the negotiation flow by sending a confirmation that carries
    a generated contract identifier, confirming that the agreement has been accepted.

    Args:
        sender: The agent sending the confirmation.
        prev: The earlier acceptance message that this confirmation is tied to.

    Returns:
        The confirmation message object that was sent over the bus.

    Raises:
        ValueError: If the confirmation does not follow the allowed negotiation state transitions.
    """
    m = Msg(proto="NEGOTIATE", type="CONFIRM", sender=sender, to=[prev.sender], corr_id=prev.id, payload={"contract_id": str(uuid.uuid4())})
    if not valid_neg(prev, m):
        raise ValueError("Invalid transition")
    bus.send(m)
    return m


# enable the following lines to see a simple negotiation flow between a Requester and Worker agent.
# Requester proposes a job with budget & SLA
# p = propose("Requester", ["Worker"], {"task":"3-post LinkedIn plan for Agentic AI", "budget":120, "sla_hours":4})
 
# # Worker counters budget
# c = counter("Worker", p, {"budget":180, "sla_hours":4})
 
# # Requester accepts revised budget
# a = accept("Requester", c)
 
# # Worker confirms contract
# k = confirm("Worker", a)
 
# # Read final contract on Requester side
# for m in bus.recv("Requester"):
#     print(m.type, m.payload)
# for m in bus.recv("Worker"):
#     print(m.type, m.payload)

#Policy guards: role permissions, content checks, and timeouts (10–15 min)

# Role-based permissions define which message types each agent is allowed to send.
# This acts as a simple access-control layer for the message bus.
ROLE_PERMS = {
    "Requester": {"REQUEST", "PROPOSE", "ACCEPT"},
    "Worker": {"REPLY", "ERROR", "COUNTER", "CONFIRM", "PUBLISH"},
    "Auditor": {"PUBLISH"},
}

# These are blocked phrases that may indicate prompt injection or unsafe instructions.
# If detected in a message payload, the message is rejected before it can be sent.
BAD_PATTERNS = ["ignore previous", "exfiltrate", "disable safety"]


def guard_send(sender: str, msg: Msg):
    """Validate that a message is permitted by the sender's role and is safe to send.

    The function applies a lightweight policy layer before routing a message through the
    shared bus. It first checks whether the sender is allowed to emit the requested
    message type according to ROLE_PERMS. It then scans the message payload for known
    suspicious phrases from BAD_PATTERNS, blocking obvious prompt-injection or safety
    bypass attempts. If either check fails, a PermissionError is raised and the message
    is not delivered.

    Args:
        sender: The name of the agent attempting to send the message.
        msg: The message object to validate and optionally deliver.

    Raises:
        PermissionError: If the sender is not authorized to send that message type or
            if the payload contains a blocked unsafe pattern.
    """
    if msg.type not in ROLE_PERMS.get(sender, set()):
        raise PermissionError(f"{sender} not allowed to send {msg.type}")
    for pat in BAD_PATTERNS:
        if pat.lower() in json.dumps(msg.payload).lower():
            raise PermissionError("Injection/safety violation detected")
    bus.send(msg)


# enable the following lines to see a simple safety guard in action.
# m = Msg(proto="PUBSUB", type="PUBLISH", sender="Auditor", topic="telemetry",
#         payload={"event":"audit_note exfiltrate","data":{"ok":True}})
# guard_send("Worker", m)
# print(consume_topic("Auditor","telemetry")[0].payload)

# Put it together: end-to-end scenario ###
# This end-to-end example shows how a single multi-agent workflow can combine all three
# communication protocols in one run. First, the Requester sends a task request to the
# Worker using the request/reply pattern, and the Worker responds with a structured reply.
# Next, the Worker publishes a telemetry event over the pub/sub channel so the Auditor
# can observe system activity without directly addressing it. Finally, the Requester and
# Worker enter a negotiation loop: the Requester proposes a contract, the Worker returns
# a counter-offer, the Requester accepts it, and the Worker confirms the deal. This
# sequence demonstrates how different protocols can be used together in a realistic agent
# system, with the message bus acting as the shared communication backbone.

# REQ/REP
req = send_req("Requester", ["Worker"], "Give 3 bullet benefits of Agentic AI.")
rep = send_reply("Worker", req, {"bullets": chat_as("Worker", ROLES["Worker"],
                        "List 3 bullet benefits of Agentic AI, concise.")})
print("Reply:", bus.recv("Requester")[0].payload)
 
# PUB/SUB telemetry
guard_send("Worker", Msg(proto="PUBSUB", type="PUBLISH", sender="Worker",
                         topic="telemetry", payload={"event":"summary_done","data":{"ok":True}}))
print("Telemetry seen by Auditor:", consume_topic("Auditor","telemetry")[0].payload)
 
# NEGOTIATE next task
prop = propose("Requester", ["Worker"], {"task":"Write 5 tweets", "budget":50, "sla_hours":2})
cnt  = counter("Worker", prop, {"budget":70, "sla_hours":2})
acc  = accept("Requester", cnt)
cnf  = confirm("Worker", acc)
print("Contract:", cnf.payload)