import heapq
import json
import re
from collections import Counter, defaultdict

_CHUNK = re.compile(r"\s+|\S+")


class Tokenizer:
    def __init__(self):
        self.vocab: dict[int, str] = {}
        self.merges: list[tuple[int, int]] = []
        self._char_to_id: dict[str, int] = {}
        self._rank: dict[tuple[int, int], int] = {}
        self._base = 0
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, num_merges: int) -> None:
        chars = sorted(set(text))
        self._base = len(chars)
        self.vocab = {i: c for i, c in enumerate(chars)}
        self._char_to_id = {c: i for i, c in enumerate(chars)}

        word_counts = Counter(_CHUNK.findall(text))
        words = [[self._char_to_id[c] for c in w] for w in word_counts]
        freqs = list(word_counts.values())

        pair_counts: Counter = Counter()
        pair_words: defaultdict = defaultdict(set)
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[wi]
                pair_words[p].add(wi)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        self.merges = []
        for _ in range(num_merges):
            best = None
            while heap:
                neg_count, p = heapq.heappop(heap)
                if pair_counts.get(p) == -neg_count:
                    best = p
                    break
            if best is None:
                break

            new_id = self._base + len(self.merges)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.merges.append(best)

            changed: set = set()
            for wi in list(pair_words.pop(best, ())):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                merged, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                words[wi] = merged
                for p in zip(merged, merged[1:]):
                    pair_counts[p] += f
                    pair_words[p].add(wi)
                    changed.add(p)
            for p in changed:
                if pair_counts[p] > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
                else:
                    del pair_counts[p]

        self._rank = {pair: r for r, pair in enumerate(self.merges)}
        self._cache = {}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _CHUNK.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = [self._char_to_id[c] for c in chunk]
        while len(ids) > 1:
            best_rank, best_i = None, None
            for i in range(len(ids) - 1):
                r = self._rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            ids[best_i:best_i + 2] = [self._base + best_rank]
        self._cache[chunk] = ids
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    def save(self, path: str) -> None:
        data = {
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": [list(m) for m in self.merges],
            "base": self._base,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = {int(k): v for k, v in data["vocab"].items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._base = data["base"]
        tok._char_to_id = {c: i for i, c in tok.vocab.items() if i < tok._base}
        tok._rank = {pair: r for r, pair in enumerate(tok.merges)}
        return tok
