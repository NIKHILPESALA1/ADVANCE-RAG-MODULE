# rag_optimizer.py
"""
RAGOptimizer: plug-and-play optimizer that upgrades naive RAG outputs.
Works with retriever outputs as (ids?, texts, metadatas?) and returns:
    optimized_texts, optimized_ids (optional)
Behavior:
 - reranks by embedding similarity (if embedder available) or token-overlap
 - applies freshness boost using metadata (if provided)
 - compresses chunks extractively (selects top sentences w.r.t. query)
"""

from typing import List, Optional, Tuple, Callable
import heapq
import re
from datetime import datetime, timezone

# Utilities
def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())

def _sentence_split(text: str):
    import re as _re
    sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sents:
        sents = [text]
    return sents

def _cosine_sim(a, b):
    import math
    dot = sum(x*y for x,y in zip(a,b))
    na = (sum(x*x for x in a))**0.5
    nb = (sum(y*y for y in b))**0.5
    if na==0 or nb==0:
        return 0.0
    return dot/(na*nb)

# Freshness heuristic
def _freshness_score(meta: dict) -> float:
    if not meta:
        return 1.0
    updated = meta.get("updated_at") or meta.get("modified_at") or meta.get("last_updated")
    if not updated:
        return 1.0
    try:
        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt).days
        if age_days < 1:
            return 1.5
        if age_days < 7:
            return 1.25
        if age_days < 30:
            return 1.0
        return 0.8
    except Exception:
        return 1.0

# Extractive compressor: pick top-k sentences by token overlap
def extractive_compress(query: str, text: str, sentences_per_chunk: int = 2) -> str:
    q_tokens = set(_tokenize(query))
    sents = _sentence_split(text)
    scored = []
    for s in sents:
        s_tokens = set(_tokenize(s))
        score = len(q_tokens & s_tokens)
        length_penalty = len(s) / 1000.0
        scored.append((score - length_penalty, s))
    top = heapq.nlargest(sentences_per_chunk, scored, key=lambda x: x[0])
    if not top:
        # fallback: return first 2 sentences or the start of text
        return " ".join(sents[:sentences_per_chunk]) if sents else (text[:500] + "...")
    return " ".join([t for _, t in top])

# Main optimizer class
class RAGOptimizer:
    def __init__(self,
                 embedder_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
                 chroma_collection = None,
                 sentences_per_chunk: int = 2):
        """
        embedder_fn: callable(list[str]) -> list[list[float]]; if None, uses overlap fallback.
                     You can pass your local_model.encode (make sure it returns list/np array).
        chroma_collection: optional chroma collection to look-up metadata by id (for freshness)
        sentences_per_chunk: how many sentences to keep per chunk when compressing.
        """
        self.embedder_fn = embedder_fn
        self.chroma = chroma_collection
        self.sentences_per_chunk = sentences_per_chunk

    def optimize(self,
                 query: str,
                 retrieved_texts: List[str],
                 retrieved_ids: Optional[List[str]] = None,
                 retrieved_metadatas: Optional[List[dict]] = None,
                 n_top: int = 5
                 ) -> Tuple[List[str], Optional[List[str]]]:
        """
        Returns (optimized_texts, optimized_ids)
        - retrieved_texts: list[str] (required)
        - retrieved_ids: optional list[str] aligned with texts
        - retrieved_metadatas: optional list[dict] aligned with texts (helps freshness)
        """
        if not retrieved_texts:
            return [], None

        # compute embeddings if available
        embeddings = None
        q_emb = None
        if self.embedder_fn is not None:
            try:
                embeddings = self.embedder_fn(retrieved_texts)
                # ensure list form
                embeddings = [list(e) for e in embeddings]
                q_emb = list(self.embedder_fn([query])[0])
            except Exception:
                embeddings = None
                q_emb = None

        scores = []
        if embeddings and q_emb is not None:
            for i, emb in enumerate(embeddings):
                sim = _cosine_sim(q_emb, emb)
                freshness = 1.0
                if retrieved_metadatas and i < len(retrieved_metadatas):
                    freshness = _freshness_score(retrieved_metadatas[i] or {})
                elif retrieved_ids and self.chroma:
                    # try to get metadata from chroma if available
                    try:
                        res = self.chroma.get(ids=[retrieved_ids[i]])
                        meta = None
                        if isinstance(res, dict) and "metadatas" in res:
                            meta_list = res["metadatas"]
                            if meta_list and len(meta_list)>0:
                                meta = meta_list[0]
                        if meta:
                            freshness = _freshness_score(meta)
                    except Exception:
                        freshness = 1.0
                scores.append((sim * freshness, i))
        else:
            # fallback: token overlap scoring
            q_tokens = set(_tokenize(query))
            for i, txt in enumerate(retrieved_texts):
                score = len(q_tokens & set(_tokenize(txt)))
                score = score - (len(txt) / 10000.0)
                freshness = 1.0
                if retrieved_metadatas and i < len(retrieved_metadatas):
                    freshness = _freshness_score(retrieved_metadatas[i] or {})
                scores.append((score * freshness, i))

        # pick top indices
        top = sorted(scores, key=lambda x: -x[0])[:max(1, n_top)]
        top_idxs = [i for _, i in top]

        optimized_texts = []
        optimized_ids = [] if retrieved_ids else None
        for idx in top_idxs:
            txt = retrieved_texts[idx]
            compressed = extractive_compress(query, txt, sentences_per_chunk=self.sentences_per_chunk)
            optimized_texts.append(compressed)
            if retrieved_ids:
                optimized_ids.append(retrieved_ids[idx])

        return optimized_texts, optimized_ids

    def make_context(self, optimized_texts: List[str], optimized_ids: Optional[List[str]] = None) -> str:
        """
        Return joined context string; includes ids for citation if provided.
        """
        if optimized_ids:
            pieces = [f"[{id_}] {txt}" for id_, txt in zip(optimized_ids, optimized_texts)]
        else:
            pieces = optimized_texts
        return "\n\n".join(pieces)
