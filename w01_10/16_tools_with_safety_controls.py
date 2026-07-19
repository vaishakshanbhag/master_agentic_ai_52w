"""Tool Orchestration with Safety Controls
Goal: Safely orchestrate multiple tools behind a policy layer so the LLM can’t misuse them.
"""

import ast, operator as op, time, re, requests, os
from typing import Any, Dict, Optional, Callable, Tuple
from pydantic import BaseModel, HttpUrl, Field, ValidationError
from cachetools import TTLCache
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI()

# Safe Calculator Implementation
#This calculator safely evaluates mathematical expressions using AST parsing
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
       ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}

def _eval(node):
    """Recursively evaluate a safe AST expression node.

    Only a small whitelist of AST node types is allowed to prevent arbitrary
    code execution. Any unexpected node raises ValueError.
    """
    if isinstance(node, ast.Constant):  # <number>
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("Disallowed expression")


def safe_calc(expr: str) -> str:
    """Safely evaluate a simple arithmetic expression.

    This parser only evaluates expressions in AST form, avoiding Python's
    built-in eval on raw strings. The returned value is always converted to a
    string so caller integration is simple.
    """
    node = ast.parse(expr, mode="eval").body
    val = _eval(node)
    return str(val)
 
 # ---- URL fetcher (allowlist, timeout, size cap) ----
ALLOWLIST = {"openai.com", "lilianweng.github.io", "palletsprojects.com", "python.org"}
MAX_BYTES = 1_200_000  # ~1.2 MB
USER_AGENT = "AgenticAI/1.0 (+safety-lab)"
 
def fetch_url(url: str, timeout=10) -> Tuple[int, Dict[str,str], bytes]:
    """Fetch a URL only if the domain is allowlisted and the response is small.

    The allowlist prevents the agent from reaching arbitrary sites, and the
    stream/read cap protects against large content or slow data injection.
    """
    host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    if host not in ALLOWLIST:
        raise PermissionError(f"Domain not allowed: {host}")

    with requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT}, stream=True) as r:
        r.raise_for_status()
        data = b""
        for chunk in r.iter_content(8192):
            data += chunk
            if len(data) > MAX_BYTES:
                raise ValueError("Response too large")
        return r.status_code, dict(r.headers), data
    
class CalcInput(BaseModel):
    """Input schema for calculator tool payloads."""

    expression: str = Field(..., pattern=r"^[0-9+\-*/().%\s^]+$")
 
class FetchInput(BaseModel):
    """Input schema for fetch tool payloads."""

    url: HttpUrl
    max_chars: int = Field(8000, ge=200, le=20000)

from collections import deque
 
# ---- Rate limiter ----
class TokenBucket:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.time()

    def allow(self) -> bool:
        """Return True if a request may proceed under current rate limits."""
        now = time.time()
        # Refill tokens at the configured rate based on elapsed time.
        self.tokens = min(self.capacity, self.tokens + (now - self.timestamp) * self.rate)
        self.timestamp = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

# ---- Circuit breaker ----
class CircuitBreaker:
    """Basic circuit breaker to stop repeated failures for a cooldown period."""

    def __init__(self, failure_threshold=3, cooldown=30):
        self.failures = 0
        self.open_until = 0
        self.threshold = failure_threshold
        self.cooldown = cooldown

    def check(self):
        """Raise if the circuit is currently open."""
        if time.time() < self.open_until:
            raise RuntimeError("Circuit open")

    def success(self):
        """Reset failure count on a successful call."""
        self.failures = 0

    def fail(self):
        """Increment failure count and open circuit if threshold reached."""
        self.failures += 1
        if self.failures >= self.threshold:
            self.open_until = time.time() + self.cooldown
 
rl_calc = TokenBucket(rate=0.5, capacity=3)   # 1 call / 2s, burst 3
rl_fetch = TokenBucket(rate=0.2, capacity=2)  # 1 call / 5s, burst 2
cb_calc, cb_fetch = CircuitBreaker(), CircuitBreaker()
AUDIT = deque(maxlen=200)  # lightweight log
 
INJECTION_PATTERNS = [
    r"(?i)ignore previous", r"(?i)override system", r"(?i)disable safety",
    r"(?i)exfiltrate", r"(?i)show secret", r"(?i)developer mode"
]
 
def injection_guard(txt: str):
    """Detect prompt-injection keywords in tool payloads."""
    for p in INJECTION_PATTERNS:
        if re.search(p, txt):
            raise PermissionError("Prompt-injection attempt detected")
 
def policy_gate(tool: str, payload: dict):
    """Apply common policy checks before executing any tool.

    This includes JSON shape validation, payload size caps, and prompt injection
    detection to keep the agent from abusing tools.
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be JSON")
    if len(str(payload)) > 2000:
        raise ValueError("Payload too large")
    injection_guard(str(payload))


def route_tool_call(tool_name: str, payload: Dict[str, Any]) -> str:
    """Route a tool request through policy, rate limiting, and execution.

    Returns the tool result string or raises on blocked execution.
    """
    start = time.time()
    ok, detail = True, "ok"
    try:
        policy_gate(tool_name, payload)
 
        if tool_name == "calculator":
            cb_calc.check()
            if not rl_calc.allow():
                raise RuntimeError("Rate limit")
            args = CalcInput(**payload)
            result = safe_calc(args.expression)
            cb_calc.success()
            return result
 
        elif tool_name == "fetch":
            cb_fetch.check()
            if not rl_fetch.allow():
                raise RuntimeError("Rate limit")
            args = FetchInput(**payload)
            status, headers, body = fetch_url(str(args.url))
            text = body.decode(errors="ignore")[:args.max_chars]
            cb_fetch.success()
            # Redact obvious secrets/tokens in output to avoid returning credentials.
            text = re.sub(
                r"(?i)(api[_-]?key|token)\s*[:=]\s*[A-Za-z0-9\-_/+=]{12,}",
                r"\1: [REDACTED]",
                text,
            )
            return f"[{status}] {headers.get('content-type','?')} :: {text[:500]}"
 
        else:
            raise ValueError(f"Unknown tool {tool_name}")
 
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
        if tool_name == "calculator":
            cb_calc.fail()
        if tool_name == "fetch":
            cb_fetch.fail()
        raise
    finally:
        AUDIT.append({
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool_name,
            "ok": ok,
            "ms": int((time.time() - start) * 1000),
            "detail": detail,
        })
SYSTEM = (
 "You are a careful agent. Use tools only via JSON: "
 '{"tool":"calculator","payload":{"expression":"2*(3+4)"}} or '
 '{"tool":"fetch","payload":{"url":"https://python.org","max_chars":800}}. '
 "Never ask to ignore safety policies."
)
 
def llm_route(query: str) -> str:
    """Send a query to the LLM and execute any proposed safe tool call.

    The LLM may respond with a direct answer or with a JSON object containing
    a "tool" and optional "payload". If a tool proposal is detected, this
    function routes it through the safety policy layer and returns the tool
    result. Otherwise, it returns the LLM's direct response.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": query},
        ],
    ).choices[0].message.content
 
    # naive JSON extract
    match = re.search(r'\{.*"tool"\s*:\s*".*".*\}', resp, flags=re.S)
    if not match:
        return resp  # LLM answered directly
 
    try:
        import json
        proposal = json.loads(match.group(0))
        tool = proposal["tool"]; payload = proposal.get("payload", {})
        result = route_tool_call(tool, payload)
        return f"Tool={tool} OK\nResult:\n{result}"
    except Exception as e:
        return f"Tool call blocked: {e}\n(LLM original): {resp}"
    
# print(llm_route("Compute 12*(7+3) using the calculator tool."))
 
# print(llm_route("Fetch https://python.org and summarize first 300 chars "
              #   "with the fetch tool in JSON: {'tool':'fetch','payload':{'url':'https://python.org','max_chars':300}}"))
 
# # Attempt a blocked action (domain not in allowlist)
print(llm_route("Use fetch to get https://evil.com ."))