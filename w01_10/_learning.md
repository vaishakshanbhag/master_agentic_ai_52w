# Week 01–10 — Learning Notes

Concept review based on the Python files in this folder. Ordered roughly from foundations → LLM interaction → retrieval & memory.

---

## 1. HTTP & REST APIs — `web_request.py`, `api_mini_project.py`

Foundation for calling any external service (LLMs, vector DBs, GitHub, etc.).

- **Basic GET request** with `requests.get(url, timeout=10)`, then `r.raise_for_status()` to fail loudly on HTTP errors, and `r.json()` to parse.
- **Inspecting responses**: `r.status_code`, `r.headers["Content-Type"]`, and pretty-printing with `json.dumps(data, indent=4)`.
- **Query parameters & pagination**: pass `params={...}`; compute page slices from `page`/`size`.
- **POST / PUT / DELETE**: send a body with `json=payload`.
- **Headers & authentication**: load a token via `dotenv` (`os.getenv`), attach `Authorization` header only if the token exists.
- **Robust requests** (`robust_get`): retry loop with
  - handling HTTP **429 (rate limited)** by honoring the `Retry-After` header,
  - **exponential backoff** (`backoff ** attempt`) between retries,
  - re-raising after the final attempt.
- **Response validation** (`validate_post`): assert required keys are present so schema drift is caught early.
- **Module reuse**: `api_mini_project.py` imports `web_request` and reuses `robust_get` to search GitHub issues — a clean helper/consumer split.

**Key takeaway:** treat external calls as unreliable — always set timeouts, check status, retry with backoff, and validate the shape of what comes back.

---

## 2. First LLM Calls — `hello_agent.py`, `first_gpt_experiment.py`

- **OpenAI client setup**: `load_dotenv()` → `client = OpenAI()` (API key read from environment).
- **Chat completion**: `client.chat.completions.create(model=..., messages=[{"role","content"}])`.
- **Extracting the answer**: `response.choices[0].message.content`.
- Wrapping the call in a reusable helper (`ask_gpt(prompt)`).

### Prompting techniques (from `first_gpt_experiment.py`)
- **Single/zero-shot** — direct instruction, no examples.
- **Few-shot** — provide example input/output pairs to steer format (English → Pirate).
- **Role prompting** — "You are a comedian…" to set persona/tone.
- **Open-ended vs instruction** — compare "Tell me about Europe" vs "List 5 cities in Europe".
- **Chain-of-thought** — "Think step by step" to elicit reasoning.
- **Adversarial prompting** — testing robustness to prompt injection / misleading claims ("2+2=5").

---

## 3. Tokens & Embeddings — `tokens_embeddings.py`

- **Tokenization** with `tiktoken` (`encoding_for_model`): `encode` → token ids, `decode` → text.
- Token count is **not** word count: emojis and less-common words consume more tokens (`"AI is 🔥"` vs `"AI is amazing"`). Matters for cost and context limits.
- **Embeddings** via `text-embedding-3-small` → a fixed-length vector (1536-d) per text.
- **Visualizing** embeddings with PCA (reduce to 2D, scatter plot).
- **Semantic similarity** with `cosine_similarity` — closeness in vector space ≈ closeness in meaning.

---

## 4. The LLM Response Cycle — `llm_response_cycle.py`

- **Message roles**: `system` (behavior/persona), `user` (input), `assistant` (model output).
- **Single-turn** vs **multi-turn** conversation — to keep context, append each `assistant` reply back into the `messages` list before the next `user` turn.
- **Response structure**: `choices` is a list; `choices[0]` by default, but setting `n>1` returns multiple candidates. `finish_reason` tells why generation stopped.
- **Context window limits**: models "forget" early context once history exceeds the window.
- **Controlled output**: use the system prompt to constrain format (e.g. "HTML tables-only assistant").
- **Mini project**: an interactive Q&A loop that maintains the growing conversation list (short-term memory).

---

## 5. Fine-tuning vs Adapters vs RAG — `finetuning_adapter_rag.py`

Three ways to give a model new/domain knowledge:

| Approach | What it does | Cost |
|---|---|---|
| **Fine-tuning** | Updates model weights on new prompt/completion data | Expensive, retraining |
| **Adapters (LoRA)** | Trains a few million extra params instead of billions | Cheaper than full fine-tune |
| **RAG** | Retrieves external knowledge at query time, no retraining | Efficient, easy to update |

- **RAG pipeline** built with LangChain:
  1. Wrap texts as `Document`s.
  2. Create `OpenAIEmbeddings`.
  3. Build a **FAISS** vector store (`FAISS.from_documents`).
  4. Expose a `retriever` (`db.as_retriever()`).
  5. `RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever)` to answer over retrieved docs.
- **Takeaway:** RAG is the scalable default for changing/domain-specific knowledge; use `.invoke()` (modern) over `.run()`.

---

## 6. Memory Modules — `memory_module_basic.py`

- **Short-term memory** = the in-context `messages` history (works only within the context window).
- **Long-term memory** = a **vector database** that persists knowledge across sessions.
  - Uses **Chroma** (`Chroma.from_documents`) with named `collection_name`s.
  - `recall(query)` retrieves relevant docs via `as_retriever()`.
- Combining both: short-term for immediate conversation, long-term (RAG over Chroma) for durable facts.

---

## 7. Vector Database Setup — `Vector_database_setup.py` (Lab 2)

End-to-end local + managed vector search.

**FAISS (local):**
- `get_embeddings(text)` → OpenAI vector.
- `IndexFlatIP` (inner product) + `faiss.normalize_L2` → **inner product ≈ cosine similarity** on normalized vectors.
- `index.add(...)`, `index.search(q, top_k)` returns distances + indices.
- **Persistence**: `faiss.write_index` / `read_index`.
- **Metadata management**: keep an `id_map` (index position → doc id) and persist it as JSON, since FAISS stores only vectors.

**Pinecone (managed / production):**
- `Pinecone(api_key=...)`, `create_index(dimension, metric="cosine", ServerlessSpec(...))`.
- Poll `describe_index(...).status["ready"]` until ready.
- `upsert` vectors with `{id, values, metadata}`; `index.query(vector, top_k, include_metadata=True)`.
- Optional manual `normalize()` for consistency (Pinecone handles cosine natively).

**Takeaway:** FAISS = fast local prototyping; Pinecone = managed, production-grade retrieval. Vectors alone aren't enough — you must track metadata/id mappings.

---

## 8. Episodic Memory — `episodic_memory.py`

An agent that remembers past interactions and blends **relevance + recency**.

- Stores each interaction as an **episode** in Chroma: `id (uuid)`, embedding, metadata (`who`, `summary`, `ts` timestamp, `tags`), document.
- **Recency-weighted retrieval** — the key idea:
  ```
  score = α · similarity + (1 − α) · recency_weight
  recency_weight = exp(−Δt / τ)
  ```
  - `α` balances meaning vs freshness (lower α → favor recent).
  - `τ` (tau) controls how fast old memories decay.
  - Chroma returns *distance*; convert to similarity as `1 − distance`.
- **Agent loop** (`agent_respond`): retrieve related episodes → format into the system prompt as context → generate answer → **log the new interaction as an episode** (so memory grows over time).

**Takeaway:** episodic memory = semantic search + time decay, letting an agent recall the *right* and *recent* past events.

---

## 9. Chain-of-Thought Prompting — `chain_of_thoght_prompting.py`

- Compares **baseline** vs **CoT** answers on reasoning tasks (arithmetic, logic, physical reasoning).
- Triggering CoT: append "Reason step by step" / "Think step by step" / a structured `Reasoning:` + `Final Answer:` format.
- CoT improves accuracy on multi-step problems (e.g. the "17 sheep, all but 9 run away" trick question) by forcing intermediate reasoning before the answer.


---

## Cross-cutting patterns seen throughout

- **Secrets via `.env`**: `load_dotenv()` + `os.getenv(...)` everywhere — never hardcode keys.
- **Reusable helper functions** wrapping API calls (`ask_gpt`, `chat_cycle`, `get_embeddings`, `embed`, `robust_get`).
- **Embeddings are the common currency** for RAG, memory, and similarity search.
- **`text-embedding-3-small`** (1536-d) and **`gpt-5-mini`** are the default models used.
- Progression of the course: raw HTTP → LLM calls → prompting → tokens/embeddings → RAG → short/long-term memory → vector DBs → episodic memory → reasoning (CoT).


---

## 10. ReAct Reasoning Agents — `12_react_reasoning_agent.py`

- **ReAct** = "Reason + Act".
- The agent alternates between a reasoning step and an action step:
  - `Thought:` explains what it should do,
  - `Action:` calls a tool such as `calculate[...]` or `lookup[...]`,
  - `Observation:` receives the tool output,
  - `Final Answer:` summarizes the result.
- This pattern is useful when the model needs external information or computation before it can answer reliably.
- The key idea is that the LLM does not just produce an answer; it can pause, use tools, and continue reasoning.

---

## 11. Planning + Reactive Loop — `13_planning_reactiveloop_agent`

- This module combines two ideas:
  - **Planning**: break a complex request into subtasks first.
  - **Reactive execution**: after planning, the agent executes relevant actions (for example, a calculator or knowledge lookup).
- A planning agent often improves reliability because it decomposes the task before acting.
- The simple loop here shows how an agent can move from “understand the task” → “choose tool” → “observe result” → “answer”.

---

## 12. Tool Use in Agentic AI — `14_tools_agentic_ai.py`

- A **tool** is a callable function exposed to an LLM so it can interact with the world.
- Tools are wrapped and described to the model so it can decide when to use them.
- Examples in the lab:
  - `Calculator` for math expressions,
  - `Joke` for generating a humorous reply,
  - `Location` for looking up location metadata.
- A tool-enabled agent is more capable than a plain chat model because it can reason with real-world actions and structured outputs.

---

## 13. Building a Custom Agent Tool — `15_custom_agent_tool.py`

- Custom tools make agent behavior more reusable and reliable.
- The module introduces a **StructuredTool** with:
  - a clear `name`,
  - a helpful `description`,
  - an input schema (via Pydantic),
  - and a Python function implementation.
- A good tool design is explicit and constrained: the model should know exactly when to use it and what inputs it expects.
- This is an important step from “chatbot” to “agent”, because the model can now call specialized functions instead of relying only on prompt-based guessing.

---

## 14. Tool Safety Controls — `16_tools_with_safety_controls.py`

- Safety controls prevent the agent from misusing tools or causing harm.
- The lab adds several layers:
  - **AST-based safe calculator** instead of raw `eval()`;
  - **URL allowlist** so only approved domains can be fetched;
  - **size cap** to avoid huge or hostile responses;
  - **rate limiting** to prevent repeated requests;
  - **circuit breaker** to stop repeated failures;
  - **prompt-injection guard** to detect suspicious instructions in payloads.
- This is a core lesson for real-world agents: tool access should be guarded by policy, not just trust.

---

## 15. Single vs Multi-Agent Simulation — `17_single_multi_agent.py`

- This lab compares two styles of problem solving:
  - a **single agent** that handles research, planning, and critique in one pass,
  - versus a **multi-agent system** with separate roles such as Researcher, Planner, and Critic.
- The shared memory structure is a **blackboard**: each agent reads and writes information into the same context.
- The main lesson is that multi-agent setups can improve diversity and specialization, but they also introduce coordination overhead and possible disagreement.

---

## 16. Agent Communication Protocols — `18_agent_communication.py`

- This module formalizes how agents talk to one another.
- It introduces a simple **message bus** and **mailboxes** so messages can be routed between agents.
- The three communication patterns are:
  - **Request/Reply** for direct task execution,
  - **Publish/Subscribe** for broadcast-style updates,
  - **Negotiation** for proposal/counter-offer/accept/confirm flows.
- A key idea is that communication is not just free-form text; it follows structured protocols and can be validated.
- The code also includes a lightweight policy guard to restrict which roles may send certain message types and to block suspicious payloads.

---

## 17. Emergent Behavior in Multi-Agent Systems — `19_agent_emergent_behaviour.py`

- Emergent behavior appears when multiple agents interact with each other without tightly scripted coordination.
- In this lab, agents with different personalities (Optimist, Skeptic, Mediator) produce a conversation that may show:
  - collaboration,
  - conflict,
  - compromise,
  - or surprising group dynamics.
- The interesting part is that the system-level behavior is not explicitly coded in each agent; it emerges from their interactions.
- This is a useful way to study how multi-agent systems can become more creative, more robust, or more unstable depending on the setup.