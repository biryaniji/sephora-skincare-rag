"""Lightweight stand-ins so rag.py can be tested without chromadb / sentence-transformers /
rank_bm25 installed. The fake embedder is a hashed bag of words, so similarity values
are NOT comparable to bge-small; tests that need the threshold patch it."""
import math
import re
import sys
import types
import zlib

import numpy as np

DIM = 512


class FakeEmbedder:
    def encode(self, text, normalize_embeddings=True, **_):
        v = np.zeros(DIM)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if len(w) > 2:
                v[zlib.crc32(w.encode()) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


def _match(meta, where):
    if where is None:
        return True
    if "$and" in where:
        return all(_match(meta, w) for w in where["$and"])
    (key, cond), = where.items()
    if isinstance(cond, dict):
        (op, val), = cond.items()
        return {"$lte": meta[key] <= val, "$gte": meta[key] >= val, "$eq": meta[key] == val}[op]
    return meta.get(key) == cond


class FakeCollection:
    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.vecs = np.array([embedder.encode(c["text"]) for c in chunks])

    def query(self, query_embeddings, n_results, where=None):
        q = np.array(query_embeddings[0])
        sims = self.vecs @ q
        idx = [i for i in np.argsort(-sims) if _match(self.chunks[i]["meta"], where)][:n_results]
        return {"ids": [[self.chunks[i]["id"] for i in idx]], "distances": [[1 - sims[i] for i in idx]]}


class _BM25Okapi:
    """Minimal Okapi BM25 (k1=1.5, b=0.75), same interface as rank_bm25."""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [{} for _ in corpus]
        for d, toks in zip(self.docs, corpus):
            for t in toks:
                d[t] = d.get(t, 0) + 1
        self.lens = [len(t) for t in corpus]
        self.avg = sum(self.lens) / len(self.lens)
        df = {}
        for d in self.docs:
            for t in d:
                df[t] = df.get(t, 0) + 1
        n = len(corpus)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def get_scores(self, query):
        out = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in query:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return np.array(out)


def install_bm25_stub():
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        sys.modules["rank_bm25"] = types.SimpleNamespace(BM25Okapi=_BM25Okapi)
