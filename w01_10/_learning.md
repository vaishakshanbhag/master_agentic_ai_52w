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
