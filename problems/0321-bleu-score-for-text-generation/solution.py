import math
import numpy as np
from collections import Counter

def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Return a Counter of all n-grams in tokens."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences.
    
    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        BLEU score between 0 and 1
    """
    # Your code here

    if not candidate or not references or max_n <= 0:
        return 0.0

    precisions = [] # p_n

    for n in range(1, max_n + 1):
        candid_ngrams = _get_ngrams(candidate, n)
        total_candidate = sum(candid_ngrams.values()) # Count_c(n_gram), denominator of p_n 
        # If there are no n-grams of this order, precision is zero.  (长度为0？)
        if total_candidate == 0:
            return 0.0
        
        # Find the matching n_grams in the reference for each candidate
        clipped_cnt = 0
        for n_gram, cnt in candid_ngrams.items():
            max_ref_count = 0
            """
            keynote-1: when there are multiple references
            """ 
            for ref in references:
                ref_ngrams = _get_ngrams(ref, n)
                max_ref_count = max(max_ref_count, ref_ngrams.get(n_gram, 0))   # 以出现次数最多的ref为准
            clipped_cnt += min(cnt, max_ref_count)   # ref 和 candid的并集

        precision = clipped_cnt / total_candidate    
        # BLEU is zero if any n-gram precision is zero.
        if precision == 0:
            return 0.0

        precisions.append(precision)

    # Brevity Penalty
    c = len(candidate)
    """
    keynote-2: find the clost reference length to the length of the candidate
    """
    ref_lengths = [len(ref) for ref in references]
    # min(iterable, key=func) -- 先对每个元素调用 func，再比较 func 的返回值
    # length只是形参，实际上传入的是 ref_lengths 中的元素
    # min 不知道也不关心形参叫什么，它只是按位置把元素传进去。
    r = min(ref_lengths, key=lambda length: (abs(length - c), length)) 
    """
    对应的"原始人"写法
    best = None
    best_score = None
    for length in ref_lengths:          # ← 这里就是“元素”
        score = (abs(length - c), length)
        if best_score is None or score < best_score:
            best_score = score
            best = length
    """

    if c >= r:
        bp = 1
    else:
        bp = math.exp(1 - r/c)
    w_logp = [math.log(prec)/max_n for prec in precisions]
    bleu = bp * math.exp(sum(w_logp))

    return bleu

