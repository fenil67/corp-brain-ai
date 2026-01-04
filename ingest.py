import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1. Load the PDF
# This reads the file page by page
print("Loading PDF...")
loader = PyPDFLoader("data/document.pdf")
raw_docs = loader.load()

# 2. Split the Text (The "Chopping Block")
# You cannot feed a 100-page PDF to Pinecone in one bite.
# We chop it into 1000-character chunks with a little overlap.
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(raw_docs)
print(f"Created {len(docs)} chunks.")

# 3. Initialize Embeddings (Must match main.py!)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 4. Upload to Pinecone
print("Uploading to the Refrigerator (Pinecone)...")
PineconeVectorStore.from_documents(
    docs,
    embeddings,
    index_name=os.getenv("PINECONE_INDEX_NAME")
)

print("Success! PDF is in the brain.")