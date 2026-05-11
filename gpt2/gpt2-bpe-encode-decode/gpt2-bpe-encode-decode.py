from typing import List, Tuple, Dict


def bpe_encode(
    text: str,
    merge_rules: List[Tuple[int, int]],
    vocab: Dict[int, bytes],
) -> List[int]:
    # Start from byte-level token IDs
    tokens = list(text.encode("utf-8"))

    # Apply merges in learned order
    for new_token_id, pair in enumerate(merge_rules, start=256):
        new_tokens = []

        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                new_tokens.append(new_token_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    return tokens


def bpe_decode(token_ids: List[int], vocab: Dict[int, bytes]) -> str:
    """
    Returns: Decoded UTF-8 string
    """
    raw_bytes = b"".join(vocab[token_id] for token_id in token_ids)

    return raw_bytes.decode("utf-8", errors="replace")