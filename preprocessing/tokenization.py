from collections import Counter

def build_vocab(series):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    chars = Counter("".join(series.dropna()))
    for c in chars:
        vocab[c] = len(vocab)
    return vocab


def tokenize(text, vocab, max_len):
    if not isinstance(text, str):
        text = ""
    tokens = [vocab.get(c, vocab["<UNK>"]) for c in text]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
    return tokens
