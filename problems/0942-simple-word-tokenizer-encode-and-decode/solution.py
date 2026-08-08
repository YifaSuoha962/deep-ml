import re

def encode(text, vocab):
    tokens = []
    for piece in re.split('([,.:;?_!"()\']|--|\s)', text):
        piece = piece.strip()
        if piece:
            tokens.append(piece)
    return [vocab[token] for token in tokens]

def decode(ids, vocab):
    # pass
    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = [inv_vocab[ind] for ind in ids]
    text = ' '.join(tokens)
    text = re.sub(r'\s+([,\.\?\!\"\(\)\'])', r'\1', text)
    return text