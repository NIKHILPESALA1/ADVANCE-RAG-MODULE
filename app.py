# run_rag_with_optimizer.py
import os
from dotenv import load_dotenv
from bm25_search import BM25Search
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from groq import Groq
from datetime import datetime, timezone
import time

from rag_optimizer import RAGOptimizer

# Load env
load_dotenv()

# Initialize local embedding model (same as your original)
local_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim

# Chroma client and collection (persistent)
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
try:
    collection = chroma_client.get_collection(collection_name)
    print(f"✅ Loaded existing collection: {collection_name}")
except Exception:
    print(f"⚠️ No existing collection found. Creating new one: {collection_name}")
    collection = chroma_client.create_collection(collection_name)

# Groq client (LLM)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Document loading & chunking ----------
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            path = os.path.join(directory_path, filename)
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
            # get file modified time as iso metadata
            mtime = os.path.getmtime(path)
            updated_at = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
            documents.append({"id": filename, "text": text, "updated_at": updated_at})
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    idx = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == L:
            break
        start = end - chunk_overlap
        idx += 1
    return chunks

# Upsert docs into chroma with chunk-level metadata and embeddings
def ingest_directory_to_chroma(directory_path):
    global collection
    docs = load_documents_from_directory(directory_path)
    print(f"Loaded {len(docs)} documents")
    all_ids = []
    all_texts = []
    all_embeddings = []
    all_metadatas = []

    for doc in docs:
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            cid = f"{doc['id']}_chunk{i+1}"
            all_ids.append(cid)
            all_texts.append(chunk)
            all_metadatas.append({"source": doc['id'], "updated_at": doc['updated_at']})
    # generate embeddings in batches
    batch_size = 64
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        batch_metas = all_metadatas[i:i+batch_size]
        print(f"Generating embeddings for batch {i//batch_size + 1} / {(len(all_texts)-1)//batch_size + 1}")
        batch_embs = local_model.encode(batch_texts, convert_to_numpy=True).tolist()
        # upsert (Chroma supports documents+embeddings+metadatas)
        try:
            collection.upsert(ids=batch_ids, documents=batch_texts, embeddings=batch_embs, metadatas=batch_metas)
        except Exception as e:
            # dimension mismatch / recreate collection fallback
            import chromadb
            if "dimension" in str(e).lower() or "dimension" in repr(e).lower():
                print("⚠️ Embedding dimension mismatch detected — recreating collection and retrying.")
                chroma_client.delete_collection(collection_name)
                collection = chroma_client.create_collection(collection_name)
                collection.upsert(ids=batch_ids, documents=batch_texts, embeddings=batch_embs, metadatas=batch_metas)
            else:
                raise e
    print("✅ All documents inserted/updated successfully into ChromaDB.")


# Build BM25 after ingestion
def build_bm25_index():
    results = collection.get(include=["documents"])
    all_ids = results["ids"]
    all_docs = results["documents"]

    # Flatten because Chroma returns a list per embedding batch
    flat_ids = []
    flat_docs = []
    for i_list, d_list in zip(all_ids, all_docs):
        flat_ids.extend(i_list)
        flat_docs.extend(d_list)

    print(f"Building BM25 index on {len(flat_docs)} chunks...")
    bm25 = BM25Search(documents=flat_docs, ids=flat_ids)
    return bm25


# Query function that returns ids, documents, metadatas
def query_documents(question, n_results=5, use_hybrid=True):
    # ----- Vector Search (Chroma) -----
    chroma_results = collection.query(query_texts=[question], n_results=n_results)
    vec_ids = chroma_results["ids"][0]
    vec_docs = chroma_results["documents"][0]
    vec_metas = chroma_results["metadatas"][0] if "metadatas" in chroma_results else [{}]*len(vec_docs)

    if not use_hybrid:
        return vec_ids, vec_docs, vec_metas

    # ----- BM25 Search -----
    bm25_hits = bm25_index.search(question, top_k=n_results)

    # Unpack BM25
    bm25_ids = [hit[0] for hit in bm25_hits]
    bm25_docs = [hit[1] for hit in bm25_hits]

    # ----- Merge (remove duplicates) -----
    merged_ids = []
    merged_docs = []
    merged_metas = []

    seen = set()

    # vector first (higher semantic weight)
    for i, d, m in zip(vec_ids, vec_docs, vec_metas):
        if i not in seen:
            merged_ids.append(i)
            merged_docs.append(d)
            merged_metas.append(m)
            seen.add(i)

    # then bm25 hits
    for i, d in zip(bm25_ids, bm25_docs):
        if i not in seen:
            merged_ids.append(i)
            merged_docs.append(d)
            merged_metas.append({})  # BM25 doesn't store metadata
            seen.add(i)

    return merged_ids, merged_docs, merged_metas


# LLM generation (use context pieces -> prompt)
def generate_response(question, context_pieces):
    context = "\n\n".join(context_pieces)
    prompt = (
        "You are a helpful assistant. Use the context below to answer concisely and cite sources using bracketed ids.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer (concise):"
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# ---------- Main flow ----------
if __name__ == "__main__":
    # 1) ingest (run once or whenever files change)
    ingest_directory_to_chroma("./news_articles")
    bm25_index = build_bm25_index()

    # 2) instantiate optimizer with your local embedder
    # Note: local_model.encode returns numpy arrays; we pass a wrapper to output lists
    embedder_fn = lambda texts: local_model.encode(texts, convert_to_numpy=True).tolist()
    optimizer = RAGOptimizer(embedder_fn=embedder_fn, chroma_collection=collection, sentences_per_chunk=2)

    # 3) sample query
    question = "How is Google competing with OPENAI. "
    # your original naive retriever call (but now returns ids/docs/metadatas)
    retrieved_ids, retrieved_texts, retrieved_metadatas = query_documents(question, n_results=8)

    # If you want pure naive behavior, comment out the optimizer usage below
    # --------- To use optimizer (Advanced RAG) ----------
    optimized_texts, optimized_ids = optimizer.optimize(
        query=question,
        retrieved_texts=retrieved_texts,
        retrieved_ids=retrieved_ids,
        retrieved_metadatas=retrieved_metadatas,
        n_top=5
    )
    context = optimizer.make_context(optimized_texts, optimized_ids)
    answer = generate_response(question, optimized_texts)
    print("\n🧠 Advanced RAG Answer (with optimizer):\n", answer)

    # --------- To revert to naive RAG, just use retrieved_texts directly ----------
    naive_answer = generate_response(question, retrieved_texts)
    print("\n🧠 Naive RAG Answer (no optimizer):\n", naive_answer)
