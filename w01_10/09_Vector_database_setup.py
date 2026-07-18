# Lab 2: Vector Database Setup with FAISS and Pinecone
# Goal: Create embeddings, index them with FAISS locally, then deploy to a managed vector DB (Pinecone) for production-grade retrieval.
# Estimated Time: 90–120 minutes
# Deliverable: Scripts/notebook that build, query, and persist a FAISS index; plus a working Pinecone index with upsert/query.

# Sample corpus
docs = [{"id":"d1","text":"Agentic AI agents use tools, memory, and goals to act."},
    {"id":"d2","text":"LangChain and CrewAI help orchestrate multi-agent workflows."},
    {"id":"d3","text":"RAG retrieves external knowledge to improve answer accuracy."},
    {"id":"d4","text":"Vector databases enable fast similarity search over embeddings."},
    {"id":"d5","text":"Planning loops and ReAct improve reasoning in complex tasks."},
]
queries = [
    "How do agents use memory?",
    "Name a framework for multi-agent orchestration.",
    "Why is RAG useful?"
]


import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI()

EMB_MODEL = "text-embedding-3-small"  # good, fast, 1536-d
# this function takes in a string and returns its embedding vector using the OpenAI API.
#  We will use this to convert our documents and queries into vectors that can be indexed
#  and searched in our vector database.
def get_embeddings(text: str):
    return client.embeddings.create(input=text, model=EMB_MODEL).data[0].embedding 
# docuument embeddings 
doc_emb= [get_embeddings(doc["text"]) for doc in docs] 
# query embeddings
query_emb = [get_embeddings(query) for query in queries]

dim= len(doc_emb[0])  # dimension of the embedding vectors
print(f"Embedding dimension: {dim}")

# Step 2: Build a FAISS index locally

import faiss
import numpy as np  

doc_emb_np = np.array(doc_emb, dtype= "float32") # convert to numpy array
index = faiss.IndexFlatIP(dim)  # inner product (use normalized vectors for cosine)
# Normalize to make IP ≈ cosine
faiss.normalize_L2(doc_emb_np)
index.add(doc_emb_np)  # add document embeddings to the index
print(f"FAISS index built with {index.ntotal} vectors.")
# # Step 3: Query the FAISS index
# def search_faiss(query_vec, top_k=3):
#     q= np.array([query_vec], dtype="float32")  # shape (1, dim)
#     faiss.normalize_L2(q)  # normalize query vector 
#     d, i = index.search(q, top_k)  # search for top_k nearest neighbors
#     return d[0], i[0]  # return indices and distances

# for qi , qv in enumerate(query_emb): 
#     d,i = search_faiss(qv, top_k=2)
#     print(f"\nQuery: {queries[qi]}")
#     for rank, (dist, idx) in enumerate(zip(d,i)):
#         print(f"  {rank}. id={docs[idx]['id']} score={round(float(dist),4)}  text={docs[idx]['text']}")

#step 4: Persist the FAISS index to disk
faiss.write_index(index, "storage/faiss_index.index") # path to save the index is "faiss_index.index"
index2 = faiss.read_index("storage/faiss_index.index")
print(f"FAISS index persisted and loaded. Total vectors: {index2.ntotal}")
#step 5: metadata management (mapping doc ids to index positions)
id_map= {i: doc["id"] for i, doc in enumerate(docs)}  # index position to doc id
# persit id_map to disk as json
import json
with open("storage/id_map.json", "w") as f:
    json.dump(id_map, f)    

# Setup[ Pinecone](https://www.pinecone.io/) - a managed vector database service
# step 1:  create index, upsert, query (managed)

import os,time, numpy as np
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX", "agentic-ai-lab") # index name from env or default

if index_name not in [i["name"]for i in pc.list_indexes()]:
    pc.create_index(name=index_name, 
                    dimension=dim, 
                    metric="cosine", 
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
    while True:
        d = pc.describe_index(index_name)
        if d.status["ready"]: break
        time.sleep(2)
index = pc.Index(index_name) # data type of index is Pinecone Index
print (f"Pinecone index '{index}' is ready.")
# # step 2: upsert documents into Pinecone
# # normalize for cosine optional; Pinecone handles cosine without normalization,
# # but normalization can be fine if you do it consistently.
def normalize(v):
    v = np.array(v, dtype="float32")
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()
 
# vectors = [{
#     "id": d["id"],
#     "values": normalize(vec),
#     "metadata": {"text": d["text"]}
# } for d, vec in zip(docs, doc_emb)]
 
# index.upsert(vectors=vectors)

# step 3: query Pinecone index
def pinecone_search(query_vec, top_k=3):
    res = index.query(
        vector=normalize(query_vec),
        top_k=top_k,
        include_metadata=True
    )
    return res
 
for qi, qv in enumerate(query_emb):
    res = pinecone_search(qv, top_k=2)
    print("\n[Query]", queries[qi])
    for match in res["matches"]:
        print(f" pconeid={match['id']}  score={round(match['score'],4)}  text={match['metadata']['text']}")

# # LangChain retriever wired to Pinecone (unable to test due to version conflicts, 
# # but this is how it would look)

# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
 
# lc_embeddings = OpenAIEmbeddings(model=EMB_MODEL)
# store = PineconeVectorStore(index_name=index_name, embedding=lc_embeddings)
 
# docs_found = store.similarity_search("How do agents use memory?", k=2)
# for d in docs_found:
#     print("-", d.page_content)