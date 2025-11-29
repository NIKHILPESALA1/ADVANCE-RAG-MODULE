# 🧠 Advanced RAG Optimization Module

This project upgrades any **Naive RAG (Retrieval-Augmented Generation) pipeline** into a **modular, plug-and-play Advanced RAG system** with:

* Hybrid Retrieval (Vector + BM25)
* Freshness-aware scoring
* Chunk compression
* Reranking
* Grounding improvements
* Plug-and-play design (attach/detach module anytime)

---

## 🚀 Project Overview

This repository contains:

* A **naive RAG pipeline** using ChromaDB + local embeddings.
* A **RAG Optimizer module** that transforms naive RAG into a production-ready RAG.
* **Hybrid retrieval** implementation using BM25 + Vector Search.

The optimizer can be connected and disconnected without modifying your existing RAG logic.

---

## 📂 Folder Structure

```
RAG/
│── app.py                   # Main RAG pipeline
│── rag_optimizer.py         # Advanced RAG optimization module
│── bm25_search.py           # BM25 keyword search engine
│── news_articles/           # Knowledge base files
│── chroma_persistent_storage/  # Persistent ChromaDB storage
│── venv/                    # Virtual environment
│── README.md
```

---

## ⚙️ Features

### ✅ Naive RAG

* Uses MiniLM embeddings
* Simple vector similarity search with ChromaDB
* Direct context → LLM answering

### 🔥 Advanced RAG (Optimizer Enabled)

* Hybrid Retrieval (BM25 + Vector)
* Extractive Chunk Compression
* Embedding-based reranking
* Metadata-based freshness scoring
* Clean API: `optimizer.optimize(query, chunks)`
* Instant plug-and-play

### 🔌 Toggle Between Modes

**Naive Mode:**

```python
answer = generate_response(question, retrieved_texts)
```

**Advanced RAG Mode:**

```python
optimized, ids = optimizer.optimize(...)
answer = generate_response(question, optimized)
```

---

## 🏗️ Setup Instructions

### 1. Clone Repository

```bash
git clone <repo-url>
cd RAG
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install rank_bm25
```

### 4. Add Environment Variables

Create `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Running the Project

```bash
python app.py
```

You will see:

* Ingestion logs
* Hybrid retrieval logs
* Advanced RAG answer
* Naive RAG answer

---

## 🧪 Testing

Try:

```
Who is Nikhil?
Difference between VIT Chennai and VIT Vellore
What does the article say about X?
```

Hybrid retrieval should return better results than vector-only RAG.

---

## 🔮 Next Enhancements (Roadmap)

* Cross-Encoder Reranker
* LLM-based compression
* Hallucination Verifier
* Auto-ingestion watcher
* Semantic chunking

---

## 👨‍💻 Author

**Nikhil Pesala**

* GitHub: [https://github.com/NIKHILPESALA1](https://github.com/NIKHILPESALA1)
* LinkedIn: [https://www.linkedin.com/in/nikhilpesala/](https://www.linkedin.com/in/nikhilpesala/)



