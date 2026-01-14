import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
# NEW IMPORTS (version-compatible helpers)
from typing import List, Dict, Any
from langchain_core.runnables import RunnablePassthrough

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fenil Patel | AI Portfolio", page_icon="👨‍💻")
st.title("👨‍💻 Chat with Fenil's Resume")
st.caption("This AI Agent has read my resume and can answer questions about my skills, experience, and goals.")


# --- 1. SETUP THE BRAIN (Cached so it runs once) ---
@st.cache_resource
def get_rag_chain():
    # Load Secrets (Streamlit Cloud uses st.secrets, Local uses os.getenv)
    # We use a try/except block to handle both environments
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
    except:
        # Fallback for local testing if .env is loaded
        from dotenv import load_dotenv
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    # Initialize Gemini
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    # Initialize Vector Store
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # --- CUSTOM CHAINS ---
    # (We paste the helper classes directly here to keep it self-contained)
    
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
                docs = self.retriever.invoke(question)
                answer = self.doc_chain.invoke({"input": question, "context": docs})
                return {"answer": answer, "context": docs}
        return _RetrievalChain(retriever, document_chain)

    # Define the Prompt
    template = """
    Answer the question based ONLY on the following context:
    {context}

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Build the Chain
    document_chain = create_stuff_documents_chain(chat_model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# Initialize the chain
rag_chain = get_rag_chain()

# --- 2. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. UI LOGIC ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask: What is Fenil's experience with Python?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # DIRECT CALL (No more requests.post)
            response = rag_chain.invoke({"input": prompt})
            
            answer = response["answer"]
            sources = [doc.page_content[:200] + "..." for doc in response["context"]]
            
            message_placeholder.markdown(answer)
            
            with st.expander("View Source Documents"):
                for i, source in enumerate(sources):
                    st.info(f"Source {i+1}: {source}")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            message_placeholder.error(f"Error: {e}")