import numpy as np
import math

PADDING_TOKEN = '[PAD]'
END_TOKEN = '[EOS]'
UNKNOWN_TOKEN = '[UNK]'


# Embedding dimension
d_model = 128
# Vocab size
V = 7816


def gen_positional_matrix(seq_len):
    positional_matrix = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(int(d_model / 2)):
            div_term = 10000 ** (2 * i / d_model)
            positional_matrix[pos][2*i] = math.sin(pos/div_term)
            positional_matrix[pos][2*i + 1] = math.cos(pos/div_term)
    return positional_matrix


def gen_ids(tokens):
    token_to_id = {}
    id_to_token = {}
    id_counter = 0
    for (token, count) in tokens:
        token_to_id[token] = id_counter
        id_to_token[id_counter] = token
        id_counter += 1
    token_to_id[PADDING_TOKEN] = id_counter
    id_to_token[id_counter] = PADDING_TOKEN
    id_counter += 1
    token_to_id[END_TOKEN] = id_counter
    id_to_token[id_counter] = END_TOKEN
    id_counter += 1
    token_to_id[UNKNOWN_TOKEN] = id_counter
    id_to_token[id_counter] = UNKNOWN_TOKEN
    return token_to_id, id_to_token


class EmbeddingLayer():
    def __init__(self, vocab_size, seq_len, d_model=128, lr=0.01):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.lr = lr
        self.seq_len = seq_len
        self.embedding_matrix = np.random.rand(vocab_size, d_model)
        self.positional_matrix = gen_positional_matrix(seq_len=seq_len)

    def gen_token_mapping(self, tokens):
        self.token_to_id = {}
        self.id_to_token = {}
        id_counter = 0
        for (token, count) in tokens:
            self.token_to_id[token] = id_counter
            self.id_to_token[id_counter] = token
            id_counter += 1
        self.token_to_id[PADDING_TOKEN] = id_counter
        self.id_to_token[id_counter] = PADDING_TOKEN
        id_counter += 1
        self.token_to_id[END_TOKEN] = id_counter
        self.id_to_token[id_counter] = END_TOKEN
        id_counter += 1
        self.token_to_id[UNKNOWN_TOKEN] = id_counter
        self.id_to_token[id_counter] = UNKNOWN_TOKEN

    def forward(self, seq):
        embeddings = np.zeros((self.seq_len, self.d_model))
        for pos, token in enumerate(seq):
            if token not in self.token_to_id:
                id = self.token_to_id[UNKNOWN_TOKEN]
            else:
                id = self.token_to_id[token]
            embeddings[pos] = self.embedding_matrix[id] + self.positional_matrix[pos]
        return embeddings

    def backwards(self, grad, tokens):
        for i, token in enumerate(tokens):
            if token not in self.token_to_id:
                id = self.token_to_id[UNKNOWN_TOKEN]
            else:
                id = self.token_to_id[token]
            self.embedding_matrix[id] -= self.lr * grad[i]
