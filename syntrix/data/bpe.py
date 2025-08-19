from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple


def get_pairs(tokens: List[str]) -> Counter:
    pairs = Counter()
    prev = None
    for t in tokens:
        if prev is not None:
            pairs[(prev, t)] += 1
        prev = t
    return pairs


class BPETokenizer:
    """Tiny BPE tokenizer (byte/char-level) for small corpora.

    Trains merges greedily to reach target vocab size.
    """

    def __init__(
        self,
        text: str | None = None,
        vocab_size: int = 256,
        vocab_tokens: List[str] | None = None,
    ):
        # If explicit vocab is provided, initialize directly
        if vocab_tokens is not None:
            self.stoi = {tok: i for i, tok in enumerate(vocab_tokens)}
            self.itos = {i: tok for tok, i in self.stoi.items()}
            self.vocab_size = len(self.stoi)
            self.merges = {}
            return
        # start from character-level symbols
        assert text is not None, "text must be provided when vocab_tokens is None"
        tokens = list(text)
        vocab: Dict[Tuple[str, ...], int] = {}
        # initialize with all single characters
        symbols = sorted(set(tokens))
        for i, ch in enumerate(symbols):
            vocab[(ch,)] = i
        # add end-of-word token (optional). Here we skip for simplicity.

        merges: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        current_tokens = [(t,) for t in tokens]

        while len(vocab) < vocab_size:
            # count pairs in stream
            pairs = Counter()
            prev = None
            for t in current_tokens:
                for s in t:
                    if prev is not None:
                        pairs[(prev, s)] += 1
                    prev = s
            if not pairs:
                break
            (a, b), _ = pairs.most_common(1)[0]
            new_sym = (a + b,)
            if new_sym in vocab:
                break
            vocab[new_sym] = len(vocab)
            merges[(a, b)] = new_sym

            # merge in stream
            new_stream = []
            i = 0
            flat = []
            for t in current_tokens:
                flat.extend(t)
            i = 0
            while i < len(flat):
                if i + 1 < len(flat) and (flat[i], flat[i + 1]) in merges:
                    new_stream.append(merges[(flat[i], flat[i + 1])])
                    i += 2
                else:
                    new_stream.append((flat[i],))
                    i += 1
            # compact consecutive tuples
            current_tokens = []
            for tup in new_stream:
                if current_tokens and len(current_tokens[-1]) == 1 and len(tup) == 1:
                    current_tokens[-1] = (current_tokens[-1][0] + tup[0],)
                else:
                    current_tokens.append(tup)

        # finalize stoi/itos from vocab
        self.itos: Dict[int, str] = {}
        self.stoi: Dict[str, int] = {}
        # ensure unique ids
        next_id = 0
        for sym_tuple in sorted(vocab.keys(), key=lambda x: (len(x[0]), x[0])):
            token = sym_tuple[0]
            if token not in self.stoi:
                self.stoi[token] = next_id
                self.itos[next_id] = token
                next_id += 1

        self.vocab_size = len(self.stoi)
        self.merges = merges

    def encode(self, s: str) -> List[int]:
        # greedy longest-match splitting using learned tokens
        result: List[int] = []
        i = 0
        while i < len(s):
            # find the longest token starting at i
            j = min(len(s), i + max(len(tok) for tok in self.stoi.keys()))
            matched = None
            while j > i:
                sub = s[i:j]
                if sub in self.stoi:
                    matched = sub
                    break
                j -= 1
            if matched is None:
                # fallback to single char
                matched = s[i]
            result.append(self.stoi[matched])
            i += len(matched)
        return result

    def decode(self, ids: List[int]) -> str:
        chars = []
        for i in ids:
            tok = self.itos.get(i, "")
            chars.append(tok)
        return "".join(chars)
