import numpy as np
from embedding import clean_text, get_sequences, count_tokens, filter_n, sort_tokens_by_count, tokenize, V, UNKNOWN_TOKEN
from model import SimpleTransformer, CategoricalCrossEntropyLoss

# AVG sentence length in english
SEQ_LEN = 16


def shift_and_one_hot(y, vocab_size, token_mapping):
    shifted_y = np.roll(y, shift=-1, axis=0)
    ids = np.array([token_mapping[token] if token in token_mapping else token_mapping[UNKNOWN_TOKEN]
                   for token in shifted_y])
    one_hot_y = np.eye(vocab_size)[ids]
    return one_hot_y

with open('./shakespeare.txt') as f:
    text = f.read()
    text = clean_text(text)
    tokens = tokenize(text)
    seqs = get_sequences(tokens, SEQ_LEN)
    counts = count_tokens(tokens)
    counts = filter_n(counts)
    sorted_tokens = sort_tokens_by_count(counts)


model = SimpleTransformer(vocab_size=V, tokens=sorted_tokens, seq_len=SEQ_LEN)
loss_func = CategoricalCrossEntropyLoss()
token_mapping = model.get_token_mapping()

for epoch in range(2):
    epoch_loss = 0
    for seq in seqs:
        out = model.forward(seq=seq)
        y_true = shift_and_one_hot(seq, V, token_mapping=token_mapping)
        loss = loss_func.calc_loss(output=out, y_true=y_true)
        epoch_loss += loss
        grad = loss_func.get_grad()
        model.backwards(grad)

    print(f"Avg epoch loss: {epoch_loss / len(seqs)} for epoch: {epoch}")