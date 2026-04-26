# implement episodic memory for a simple agent. 

import time, uuid,math, os
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
import chromadb 

load_dotenv()
client = OpenAI()
EMB_MODEL = "text-embedding-3-small"  # good, fast, 1536-d

def embed(text: str):
    return client.embeddings.create(input=text, model=EMB_MODEL).data[0].embedding

 # create a local ChromaDB collection for episodic memory
chroma_client = chromadb.Client()
episodes = chroma_client.get_or_create_collection(name="episodic_memory") # id, embedding, metadata (timestamp, type, content)

def now_ts():

    return int(time.time())

def add_episode(summary:str, who:str="user", tags=None): 
    # sore and episode with text summary + metadata
    eid= str(uuid.uuid4())  # unique episode id
    meta={
        "who": who,
        "summmary": summary,
        "ts": now_ts(),
        "tags": "," .join(tags) if tags else ""
    }
    episodes.add( ids=[eid], embeddings=[embed(summary)], metadatas=[meta], 
                 documents=[summary])
    return eid

def ts_to_str(ts:int):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# Recency similarity scoring 
# We’ll blend embedding similarity with time decay: 
# score = α·similarity + (1–α)·recency_weight, where recency_weight = exp(-Δt/τ).
# the function searchs based on similarity and recency if you want to give more weight to recency then 
# set alpha lower (e.g., 0.5) and tau controls how quickly recency decays 
# (e.g., 72 hours).
def search_episodes(query:str, k=5, alpha=0.7, tau_hours=72):
        """Return top-k episodes by blended similarity + recency."""
        qemb= embed(query)
        #raw similary search

        res= episodes.query(query_embeddings=[qemb], n_results=20, 
                            include=["metadatas","distances", "documents", "embeddings"])
        if not res["ids"]:
            return []  # no matches
        now= now_ts()
        tau= tau_hours * 3600  # convert hours to seconds
        out = []
        for i in range(len(res["ids"][0])):
            meta = res["metadatas"][0][i]
            doc  = res["documents"][0][i]
            # Chroma returns distance; convert to similarity (cosine-ish). Guard if distance missing.
            distance = res["distances"][0][i] if res["distances"] else 0.0
            similarity = 1 - distance  # crude invert; OK for our demo
            dt = max(0, now - int(meta["ts"]))
            rec= math.exp(-dt / tau)
            blended = alpha*similarity + (1-alpha)*rec
            out.append({
            "id": res["ids"][0][i],
            "summary": doc,
            "who": meta["who"],
            "ts": meta["ts"],
            "when": ts_to_str(int(meta["ts"])),
            "similarity": round(similarity,4),
            "recency": round(rec,4),
            "score": round(blended,4),
            "tags": meta.get("tags","")
        })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:k]  

# function to handle user input, retrieve relevant episodes, 
# and generate a response using the LLM.
def llm_chat(system:str, user:str):
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ]
    )
    return resp.choices[0].message.content
 
 # function to format retrieved episodes into a readable string for the system prompt.
def format_memories(mem_list):
    lines = []
    for m in mem_list:
        lines.append(f"- ({m['when']}) [{m['who']}] {m['summary']}")
    return "\n".join(lines)

#function to handle user input, retrieve relevant episodes, 
# generate a response using the LLM, and log this interaction as a new episode.
def agent_respond(user_text:str):
    # 1) retrieve related episodes
    mems = search_episodes(user_text, k=3, alpha=0.7, tau_hours=72)
    mem_context = format_memories(mems) if mems else "None"
    system = (
        "You are an assistant with episodic memory. "
        "If 'Relevant episodes' contain useful context, use it to inform your answer. "
        "Be concise and cite facts to the extent they appear in the episodes."
        f"\n\nRelevant episodes:\n{mem_context}"
    )
    # 2) generate answer
    answer = llm_chat(system, user_text)
    # 3) log this interaction as an episode
    add_episode(summary=f"Q: {user_text} | A: {answer[:250]}...", who="agent", tags=["dialog"])
    return answer, mems


# Seed memory with some episodes (simulate prior sessions)
add_episode("User said their favorite framework is LangChain and they build agents for education.", who="user", tags=["pref","framework"])
add_episode("We integrated Pinecone yesterday to speed up semantic search.", who="agent", tags=["project","infra"])
add_episode("User asked about planning loops vs reactive loops in agents.", who="user", tags=["topic","planning"])
 
# Now ask something related
question = "Which framework did I say I like, and what did we integrate yesterday?"
answer, used_mems = agent_respond(question)
 
print("\n--- Agent Answer ---\n", answer)
print("\n--- Memories Used ---")
for m in used_mems:
    print(f"{m['when']} :: {m['summary']}  (score={m['score']})")
# agent_respond("Remind me what we discussed about planning vs reactive loops.")
# agent_respond("Do we already have a vector DB set up?")