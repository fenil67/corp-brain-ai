import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone

# 1. Load Environment Variables
load_dotenv()

# Configuration
INDEX_NAME = "knowledge-base"  # Ensure this matches your Pinecone Index Name exactly

def wipe_index():
    print(f"💣 Preparing to wipe index: '{INDEX_NAME}'...")
    
    # Check for API Key
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("❌ PINECONE_API_KEY not found in .env file")

    # 2. Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)
    
    # 3. Check current stats
    try:
        stats = index.describe_index_stats()
        count = stats.total_vector_count
        print(f"📊 Current Status: {count} vectors found.")
    except Exception as e:
        print(f"⚠️ Error connecting to index: {e}")
        return

    # 4. Delete Everything
    if count > 0:
        print("🧹 Deleting all vectors... (This may take a few seconds)")
        index.delete(delete_all=True)
        
        # 5. Wait for propagation (Crucial Step)
        print("⏳ Waiting 10 seconds for Pinecone to update...")
        time.sleep(10)
        
        # 6. Verify
        new_stats = index.describe_index_stats()
        new_count = new_stats.total_vector_count
        print(f"📉 Status after wipe: {new_count} vectors.")
        
        if new_count == 0:
            print("✅ Brain is completely empty. Ready for new ingestion.")
        else:
            print("⚠️ WARNING: Vectors still exist. Please run this script again.")
    else:
        print("✅ Index was already empty. You are good to go.")

if __name__ == "__main__":
    wipe_index()