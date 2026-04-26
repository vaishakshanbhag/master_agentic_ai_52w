# this is lab 1 focused on Building Short-Term and Long-Term Memory Modules

import os
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()
# # here we are using short-term memory by keeping the conversation history in the messages list. The model can refer back to previous messages to maintain context and provide coherent responses.
def chat_with_memory(messages):
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages
    )
    return response.choices[0].message.content

# #simulate a conversation with memory
# conversation = [
#     {"role": "system", "content": "You are a helpful tutor."},
#     {"role": "user", "content": "My name is Alex."},
#     ]
# reply1=chat_with_memmory(conversation)
# conversation.append({"role": "assistant", "content": reply1})
# conversation.append({"role": "user", "content": "What is my name? Respond in a poetic manner."})    
# reply2=chat_with_memmory(conversation)

# print("Assistant's first reply:", reply1)
# print("Assistant's second reply:", reply2)


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# # In this part, we are creating a long-term memory module using a vector database (Chroma) to store and retrieve information. We create some documents, convert them into embeddings, and store them in the vector database. The recall function allows us to query the database and retrieve relevant information based on the query.
# #Create Some Knowledge Base
docs = [
    Document(page_content="Agentic AI agents use tools and memory."),
    Document(page_content="LangChain helps build autonomous agents."),
    Document(page_content="RAG improves accuracy by retrieving knowledge."),
]
db = Chroma.from_documents(docs, embeddings, collection_name="long_term_memory")
#store in vector DB

docs2 = [
    Document(page_content="Node.js is a cross-platform, open-source JavaScript runtime environment that can run on Windows, Linux, and macOS."),
    Document(page_content="Node.js uses an event-driven, non-blocking I/O model that makes it lightweight and efficient."),
    Document(page_content="The package manager for Node.js is npm, which is the world's largest software registry."),
    Document(page_content="In Node.js, the 'fs' module allows you to work with the file system on your computer."),
    Document(page_content="The CommonJS module system is used in Node.js, where 'require' is used to import modules.")
]
db = Chroma.from_documents(docs2, embeddings, collection_name="detail_of_nodejs")
#query vector DB function
def recall(query):
    retriever = db.as_retriever()
    results = retriever.get_relevant_documents(query)
    return [r.page_content for r in results]

# print("Recall:", recall(" Node.js uses what?"))


from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

retriever = db.as_retriever()
qa= RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-5-mini"),
    retriever=retriever,
)
conversation =   [
    {"role":"system","content":"You are a teaching AI agent."},
    {"role":"user","content":"Remember my favorite framework is LangChain."},
    {"role":"assistant","content":"Got it, your favorite framework is LangChain."}
]

# user asks later
conversation.append({"role":"user","content":"What’s my favorite framework and how do agents use memory?"})


# First answer uses short-term memory
# short_term_ans = chat_with_memory(conversation)
# print("\nShort-term:", short_term_ans)
# Second answer augments with long-term memory
response = qa.invoke({"query": "How do agents use memory?"})

print("\nShort+Long-term:", response["result"])