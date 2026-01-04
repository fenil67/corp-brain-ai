import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
# NEW IMPORTS (version-compatible helpers)
from typing import List, Dict, Any
from langchain_core.runnables import RunnablePassthrough

# --- CUSTOM CHAINS (Fixed for New LangChain Versions) ---

def create_stuff_documents_chain(llm, prompt):
    class _DocChain:
        def __init__(self, llm, prompt):
            self.llm = llm
            self.prompt = prompt

        def invoke(self, inputs: Dict[str, Any]) -> str:
            docs: List[Any] = inputs["context"]
            question: str = inputs["input"]
            context_text = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
            messages = self.prompt.format_messages(context=context_text, input=question)
            response = self.llm.invoke(messages)
            return getattr(response, "content", response)

    return _DocChain(llm, prompt)

def create_retrieval_chain(retriever, document_chain):
    class _RetrievalChain:
        def __init__(self, retriever, doc_chain):
            self.retriever = retriever
            self.doc_chain = doc_chain

        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            question: str = inputs["input"]
            
            # --- THE FIX IS HERE ---
            # Old version: docs = self.retriever.get_relevant_documents(question)
            # New version:
            docs = self.retriever.invoke(question)
            
            answer = self.doc_chain.invoke({"input": question, "context": docs})
            return {"answer": answer, "context": docs}

    return _RetrievalChain(retriever, document_chain)

# --- APP SETUP ---

load_dotenv()

app = FastAPI()

# 1. Setup Brain & Memory
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Get top 2 pages

# 2. The New Chain Strategy
template = """
Answer the question based ONLY on the following context:
{context}

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(chat_model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.post("/chat")
def chat_with_resume(question: str):
    response = retrieval_chain.invoke({"input": question})
    
    # Extract sources
    sources = [doc.page_content[:200] + "..." for doc in response["context"]]
    
    return {
        "answer": response["answer"],
        "sources": sources
    }