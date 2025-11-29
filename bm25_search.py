# bm25_search.py
from rank_bm25 import BM25Okapi
import re

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

class BM25Search:
    def __init__(self, documents=None, ids=None):
        """
        documents: list[str]
        ids: list[str]
        """
        self.documents = documents or []
        self.ids = ids or []
        self.tokenized_docs = [tokenize(d) for d in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs) if self.documents else None

    def add_batch(self, docs, ids):
        self.documents.extend(docs)
        self.ids.extend(ids)
        self.tokenized_docs = [tokenize(d) for d in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 5):
        if not self.bm25:
            return []
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.ids[i], self.documents[i]) for i in ranked]
