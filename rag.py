import os
import re
import csv
from typing import List
from collections import defaultdict
from datetime import datetime
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document

from gmail_fetcher import authenticate_gmail, fetch_emails

# === Email Handling ===

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
user_interactions = defaultdict(int)


def clean_html(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw or "")


def build_email_documents() -> List[Document]:
    service = authenticate_gmail()
    raw_emails = fetch_emails(service)
    raw_emails = raw_emails[:10]
    docs = []
    for e in raw_emails:
        body = clean_html(e.get("body", ""))
        docs.append(Document(
            page_content=body,
            metadata={
                "from": e.get("from", ""),
                "subject": e.get("subject", ""),
                "timestamp": e.get("timestamp", "")
            }
        ))
    return docs


def save_emails_to_csv(email_docs: List[Document], filename='emails.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['from', 'subject', 'body', 'timestamp'])
        writer.writeheader()
        for doc in email_docs:
            writer.writerow({
                'from': doc.metadata.get('from', ''),
                'subject': doc.metadata.get('subject', ''),
                'body': doc.page_content,
                'timestamp': doc.metadata.get('timestamp', '')
            })


def load_email_documents(csv_path='emails.csv'):
    loader = CSVLoader(file_path=csv_path, metadata_columns=["from", "subject", "timestamp"])
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)  # Try smaller chunks with more overlap
    chunked_docs = splitter.split_documents(docs)
    
    return chunked_docs


# ===Chat Setup ===

# Step 1: Fetch emails and save
email_docs = build_email_documents()
save_emails_to_csv(email_docs)

# Step 2: Load and chunk
documents = load_email_documents() 

# Step 3: Embed and retrieve
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embedding_model)  
retriever = vectorstore.as_retriever()

# Step 4: Mistral chat model 
model = ChatMistralAI(model="mistral-large-latest", temperature=0.7)  

# Step 5: RAG logic
def call_model(state: MessagesState):
    user_query = state["messages"][-1].content
    relevant_docs = retriever.get_relevant_documents(user_query) 
    context = "\n\n---\n\n".join(
        f"From: {doc.metadata.get('from', '')}\nSubject: {doc.metadata.get('subject', '')}\n\n{doc.page_content}"
        for doc in relevant_docs
    )

    prompt = f"""You are an AI assistant reading a user's Gmail inbox. Use the following emails to answer the question.

Emails:
{context}

Question: {user_query}

Answer:"""

    response = model.invoke([HumanMessage(content=prompt)])
    return {"messages": response}

# Step 6: LangGraph workflow
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Step 7: Chat loop
def run_chat(thread_id="email-thread-1"):
    config = {"configurable": {"thread_id": thread_id}}
    print("\n💬 Ask me anything about your Gmail emails (type 'exit' to quit):\n")

    messages = []  # Local message history

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        messages.append(HumanMessage(content=user_input))  # Add user input to history
        output = app.invoke({"messages": messages}, config)
        messages.extend(output["messages"])  # Add assistant's reply to history
        print("\n🤖", output["messages"][-1].content)
        print("-" * 50)

if __name__ == "__main__":
    run_chat()
