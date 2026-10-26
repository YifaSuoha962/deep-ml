import torch
from collections import defaultdict


def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.

    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
        num_merges: Number of merge operations to perform.

    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
    """
    # 将语料转换为 (token 列表, 频率) 的形式
    words = [(tokens.split(), freq) for tokens, freq in corpus.items()]

    merges = []

    for _ in range(num_merges):
        # 统计所有相邻 token 对的加权出现次数
        pair_counts = defaultdict(float)

        for tokens, freq in words:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq

        if not pair_counts:
            break  # 没有可合并的对，提前停止

        # 找到出现次数最多的对；并列时保留先遇到的
        best_pair = None
        best_count = -1.0

        for tokens, freq in words:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                count = pair_counts[pair]
                if count > best_count:
                    best_count = count
                    best_pair = pair

        if best_pair is None or best_count <= 0:
            break

        a, b = best_pair
        merged_token = a + b

        # 在整个语料中合并该对
        new_words = []
        for tokens, freq in words:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == a
                    and tokens[i + 1] == b
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_words.append((new_tokens, freq))

        words = new_words
        merges.append(best_pair)

    return merges