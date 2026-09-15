import os
import glob
from typing import List, Dict, Any

# Ensure HuggingFace uses local cached model offline without network delays
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.path.join(os.path.dirname(__file__), 'chroma_db')
DOCS_DIR = os.path.join(os.path.dirname(__file__), 'docs')
COLLECTION_NAME = 'zepto_policies'

# Singleton model loader
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    return _embed_model

# -------------------------------------------------------------------------
# STRUCTURED PROMPT TEMPLATE (Role-Context-Task-Format-Length Skeleton)
# Includes explicit negative constraint and few-shot example.
# -------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are Zepto's official AI Policy Assistant (ROLE).
Your objective is to assist customers with accurate answers regarding Zepto delivery, returns, membership, cancellation, and support guidelines (TASK).

Here is the retrieved policy context:
{context}

STRICT CONSTRAINTS (NEGATIVE CONSTRAINTS):
1. Do NOT answer using information not present in the provided context.
2. Do NOT invent, extrapolate, or guess policy terms, fees, or timelines that are not explicitly documented.
3. If the context does not contain enough information to answer the question, state clearly: "I am sorry, but that information is not available in Zepto's documented policies."

OUTPUT FORMAT (FORMAT):
Respond with a single valid JSON object containing exactly three keys:
- "answer": string containing the policy answer.
- "sources": list of string document IDs (e.g., ["doc_01", "doc_03"]).
- "confidence": float between 0.0 and 1.0 representing answer certainty.

RESPONSE LENGTH (LENGTH):
Keep the answer concise, direct, and factual—between 2 and 4 sentences.

FEW-SHOT EXAMPLE:
User Question: What are the perks of Zepto Pass+?
Retrieved Context: [doc_03] Zepto offers three account tiers: Basic (free), Zepto Pass (INR 49 per month), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live).
Response:
{
  "answer": "Zepto Pass+ costs INR 99 per month and includes free priority delivery, 10% off select categories, and 24-hour early access to limited-time deals.",
  "sources": ["doc_03"],
  "confidence": 0.98
}
"""

def get_chroma_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)

def ingest_corpus():
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    if not doc_files:
        raise FileNotFoundError(f"No policy documents found in {DOCS_DIR}")
        
    ids = []
    documents = []
    metadatas = []
    
    for fpath in doc_files:
        doc_id = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        ids.append(doc_id)
        documents.append(text)
        metadatas.append({"doc_id": doc_id, "source": os.path.basename(fpath)})
        
    model = get_embed_model()
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    
    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Successfully ingested {len(ids)} documents into ChromaDB collection '{COLLECTION_NAME}'.")
    return collection

def retrieve_top_k(query: str, k: int = 3) -> List[Dict[str, Any]]:
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # If collection is empty, ingest first
    if collection.count() == 0:
        ingest_corpus()
        
    model = get_embed_model()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved = []
    if results and "ids" in results and results["ids"]:
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            doc_text = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if "distances" in results else 0.0
            # Cosine distance to similarity: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - distance))
            retrieved.append({
                "id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "similarity": similarity
            })
    return retrieved

if __name__ == "__main__":
    print("Testing Ingestion & Retrieval...")
    ingest_corpus()
    sample_query = "What happens if my order arrives with spoiled or damaged items?"
    hits = retrieve_top_k(sample_query, k=3)
    print(f"\nQuery: {sample_query}")
    for h in hits:
        print(f"  Hit: [{h['id']}] (Sim: {h['similarity']:.4f}) -> {h['text'][:100]}...")
