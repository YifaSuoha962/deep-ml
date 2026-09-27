def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    tokens = corpus.split()
    cnt_w = tokens.count(word)
    prob = cnt_w / len(tokens)
    return round(prob, 4)