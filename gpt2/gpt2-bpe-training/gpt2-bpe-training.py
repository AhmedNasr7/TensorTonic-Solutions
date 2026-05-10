from typing import Tuple, List, Dict
from collections import Counter

def bpe_train(text: str, target_vocab_size: int) -> Tuple[List[Tuple[int, int]], Dict[int, bytes]]:
    text_bytes = text.encode("utf-8")

    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    tokens = list(text_bytes)

    merge_rules: List[Tuple[int, int]] = []
    next_id = 256

    while len(vocab) < target_vocab_size:
        pair_counts = Counter(zip(tokens, tokens[1:]))

        if not pair_counts:
            break

        # highest count, then smallest pair lexicographically
        best_count = max(pair_counts.values())
        best_pair = min(
            pair for pair, count in pair_counts.items()
            if count == best_count
        )

        a, b = best_pair
        vocab[next_id] = vocab[a] + vocab[b]
        merge_rules.append(best_pair)

        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(next_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens
        next_id += 1

    return merge_rules, vocab