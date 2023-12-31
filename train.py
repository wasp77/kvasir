import numpy as np
from embedding import V, UNKNOWN_TOKEN
from model import SimpleTransformer, CategoricalCrossEntropyLoss
from dataset import Dataset

# AVG sentence length in english
SEQ_LEN = 10


def shift_and_one_hot(y, vocab_size, token_mapping):
    shifted_y = np.roll(y, shift=-1, axis=0)
    ids = np.array([token_mapping[token] if token in token_mapping else token_mapping[UNKNOWN_TOKEN]
                   for token in shifted_y])
    one_hot_y = np.eye(vocab_size)[ids]
    return one_hot_y

def encode_decode(tokens, mapping):
    return [mapping[token] for token in tokens]


if __name__ == "__main__": 

    dataset = Dataset(text_path='./single_sentence.txt', seq_len=SEQ_LEN)
    sorted_tokens = dataset.sorted_tokens
    # Need to account for the special tokens
    vocab_size = dataset.vocab_size + 3
    seqs = dataset.seqs
    print(seqs)


    model = SimpleTransformer(vocab_size=vocab_size,
                            tokens=sorted_tokens, seq_len=SEQ_LEN)
    loss_func = CategoricalCrossEntropyLoss()
    token_mapping, id_to_token = model.get_token_mapping()

    for epoch in range(1000):
        epoch_loss = 0
        for seq in seqs:
            out = model.forward(seq=seq)
            y_true = shift_and_one_hot(seq, vocab_size, token_mapping=token_mapping)
            loss = loss_func.calc_loss(output=out, y_true=y_true)
            epoch_loss += loss
            grad = loss_func.get_grad()
            model.backwards(grad)

        print(f"Avg epoch loss: {epoch_loss / len(seqs)} for epoch: {epoch}")


    context_str = 'the quick brown'
    context_tokens = context_str.split()
    enc_seq = encode_decode(context_tokens, token_mapping)
    out = model.generate(enc_seq, max_new_tokens=1)
    dec_seq = encode_decode(out, id_to_token)
    print(dec_seq)
