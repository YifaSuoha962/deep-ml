def train_tokenizer(corpus, vocab_size):
    """
    Build a BPE-style tokenizer from the training corpus.

    Args:
        corpus: list[str]
            Training strings, e.g. ["emma", "olivia", ...]
        vocab_size: int
            Maximum vocabulary size.

    Returns:
        encode: str -> list[int]
        decode: list[int] -> str
    """

    # ------------------------------------------------------------
    # 1. Initial character vocabulary
    # ------------------------------------------------------------
    chars = sorted(set(ch for text in corpus for ch in text))

    if len(chars) > vocab_size:
        raise ValueError(
            "vocab_size is smaller than the number of unique characters"
        )

    # token string -> token id
    token_to_id = {}
    # token id -> token string
    id_to_token = []

    for ch in chars:
        token_to_id[ch] = len(id_to_token)
        id_to_token.append(ch)

    # Each training string initially consists of character tokens
    sequences = [list(text) for text in corpus]

    # Learned merges, in order
    merges = []

    # ------------------------------------------------------------
    # Helper: merge one pair everywhere in a sequence
    # ------------------------------------------------------------
    def merge_pair(seq, pair, merged_token):
        a, b = pair
        result = []

        i = 0
        while i < len(seq):
            if (
                i + 1 < len(seq)
                and seq[i] == a
                and seq[i + 1] == b
            ):
                result.append(merged_token)
                i += 2
            else:
                result.append(seq[i])
                i += 1

        return result

    # ------------------------------------------------------------
    # 2. Learn BPE merges
    # ------------------------------------------------------------
    while len(id_to_token) < vocab_size:

        pair_counts = {}

        # Count adjacent pairs within each name
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        # Pick most frequent pair.
        # Lexicographic tie-break makes training deterministic.
        best_pair, best_count = min(
            pair_counts.items(),
            key=lambda item: (
                -item[1],
                item[0][0],
                item[0][1]
            )
        )

        # Do not learn one-off substrings:
        # they usually memorize individual training names
        # instead of learning reusable subwords.
        if best_count < 2:
            break

        merged_token = best_pair[0] + best_pair[1]

        # The same literal string may theoretically already exist.
        # If so, no new vocabulary item is needed, but such a case
        # is normally rare with character-initialized BPE.
        if merged_token not in token_to_id:
            token_to_id[merged_token] = len(id_to_token)
            id_to_token.append(merged_token)

        merges.append((best_pair, merged_token))

        # Apply this merge to the whole training corpus
        sequences = [
            merge_pair(seq, best_pair, merged_token)
            for seq in sequences
        ]

    # ------------------------------------------------------------
    # 3. Encoding
    # ------------------------------------------------------------
    def encode(text):
        # Start from characters
        seq = list(text)

        # All characters must be representable.
        for ch in seq:
            if ch not in token_to_id:
                raise ValueError(
                    "Encountered unseen character: {!r}".format(ch)
                )

        # Apply learned merges in exactly the order they were learned
        for pair, merged_token in merges:
            seq = merge_pair(seq, pair, merged_token)

        return [token_to_id[token] for token in seq]

    # ------------------------------------------------------------
    # 4. Decoding
    # ------------------------------------------------------------
    def decode(ids):
        return "".join(id_to_token[i] for i in ids)

    return encode, decode