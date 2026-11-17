"""Tokenization, lexical embeddings and BM25 used for hybrid retrieval and offline fallbacks."""

import hashlib
import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in into is it its "
    "may not of on or our should so such than that the their then there these they this to "
    "was we were what when where which who why will with you your".split()
)
_WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def hash_embed(text: str, dim: int = 512) -> list[float]:
    """Signed feature hashing of unigrams and bigrams, L2-normalised.

    Lexical similarity only - it is a dependency-free offline fallback, not a semantic model.
    """
    toks = tokenize(text)
    feats = Counter(toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)])
    vec = [0.0] * dim
    for feat, count in feats.items():
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
        vec[h % dim] += (1.0 if (h >> 32) & 1 else -1.0) * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.lengths, strict=True):
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 25]
