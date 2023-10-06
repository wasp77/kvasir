import re
import numpy as np
import math

PADDING_TOKEN = '[PAD]'
END_TOKEN = '[EOS]'
UNKNOWN_TOKEN = '[UNK]'

# AVG sentence length in english
SEQ_LEN = 15

# Embedding dimension
d_model = 128
# Vocab size
V = 7816

def gen_positional_matrix():
    positional_matrix = np.zeros((SEQ_LEN + 1, d_model))
    # We have a EOS token
    for pos in range(SEQ_LEN + 1):
        for i in range(int(d_model / 2)):
            div_term = 10000 ** (2 * i / d_model)
            positional_matrix[pos][2*i] = math.sin(pos/div_term)
            positional_matrix[pos][2*i + 1] = math.cos(pos/div_term)
    return positional_matrix

embedding_matrix = np.random.rand(V,d_model)
positional_matrix = gen_positional_matrix()


def clean_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert double new lines to single new lines
    text = re.sub(r'\n+', '\n', text)
    # Make everthing lowercase
    text = text.lower()
    return text

def tokenize(text):
    return text.split()

def count_tokens(tokens):
    count_map = {}
    for token in tokens:
        if token in count_map:
            count_map[token] += 1
        else:
            count_map[token] = 1
    return count_map

def filter_n(counts, n=5):
    filtered_counts = {}
    for (key, value) in counts.items():
        if value > n:
            filtered_counts[key] = value
    return filtered_counts

def sort_tokens_by_count(counts):
    tuples = list(counts.items())
    return sorted(tuples, key=lambda x: x[1], reverse=True)

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

def get_sequences(text, sequence_len=SEQ_LEN):
    sequences = []
    sequence = []
    count = 0
    for token in text:
        if count < sequence_len:
            sequence.append(token)
            count += 1
        else:
            sequence.append(END_TOKEN)
            sequences.append(sequence)
            sequence = [token]
            count = 1
    if len(sequence) < sequence_len:
        diff = sequence_len - len(sequence)
        padding = [PADDING_TOKEN] * diff
        sequence.extend(padding)
        sequence.append(END_TOKEN)
        sequences.append(sequence)
    return sequences

def embed(tokens, token_to_id):
    embeddings = np.zeros((SEQ_LEN + 1, d_model))
    for pos, token in enumerate(tokens):
        if token not in token_to_id:
            print('handling unknown')
            id = token_to_id[UNKNOWN_TOKEN]
        else:
            id = token_to_id[token]
        embeddings[pos] = embedding_matrix[id] + positional_matrix[pos]
    return embeddings


with open('./shakespeare.txt') as f:
    text = f.read()
    text = clean_text(text)
    tokens = tokenize(text)
    seq = get_sequences(tokens)
    counts = count_tokens(tokens)
    counts = filter_n(counts)
    sorted_tokens = sort_tokens_by_count(counts)
    token_to_id, id_to_token = gen_ids(sorted_tokens)
    print(embed(seq[0], token_to_id))

