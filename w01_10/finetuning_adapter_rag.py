import os
from xmlrpc import client
from dotenv import load_dotenv
from openai import OpenAI   
load_dotenv()

client = OpenAI()   


## define a simple knowledge base and questions for RAG

kb = [
    "Agentic AI agents use memory, tools, and goals to act.",
    "LangChain and CrewAI are popular frameworks for building AI agents.",
    "Retrieval-Augmented Generation (RAG) improves accuracy by fetching external knowledge."
]
questions = [
    "What are the key components of Agentic AI?",
    "Name one framework for AI agents.",
    "How does RAG improve answers?"
]

## Fine tuning = updating model weights with new data, which can be time-consuming and resource-intensive. In contrast, RAG allows us to enhance the model's performance by providing it with relevant information from an external knowledge base without needing to retrain the entire model. This makes RAG a more efficient and scalable solution for improving the accuracy of AI agents, especially when dealing with rapidly changing or domain-specific information.

# from datasets import Dataset

# train_data = Dataset.from_dict({
#     "prompt": [
#         "Q: What are the key components of Agentic AI?\nA:",
#         "Q: Name one framework for AI agents.\nA:",
#         "Q: How does RAG improve answers?\nA:"
#     ],
#     "completion": [
#         " Agentic AI agents use memory, tools, and goals to act.",
#         " LangChain is a framework for building AI agents.",
#         " RAG improves accuracy by fetching external knowledge before answering."
#     ]
# })
# print(train_data)

## Adapter LoRA

# from transformers import AutoModelForCausalLM, AutoTokenizer
 
# model_name = "distilgpt2"
# tok = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
 
# print("Base model loaded:", model_name)
# print("With LoRA/adapters, you’d only train a few million params instead of billions.")

## RAG 
# 1. Modern Partner & Core Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 2. Classic Chain Imports (Since RetrievalQA moved to classic in v1.0)
from langchain_classic.chains import RetrievalQA

# --- Build vector DB ---
# Assuming 'kb' is your list of strings
docs = [Document(page_content=x) for x in kb]
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()

# --- Setup QA Chain ---
# Note: RetrievalQA is now part of the 'classic' bundle
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-5-mini"),
    chain_type="stuff",
    retriever=retriever
)

# --- Run Queries ---
for q in questions:
    print("\nQ:", q)
    # In 2026, .invoke() is the standard over .run()
    result = qa.invoke({"query": q})
    print("A:", result["result"])