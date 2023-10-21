import re
from embedding import PADDING_TOKEN, END_TOKEN, UNKNOWN_TOKEN

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

def get_sequences(text, seq_len):
    sequences = []
    sequence = []
    count = 0
    for token in text:
        # Allow for EOS at the end
        if count < seq_len - 1:
            sequence.append(token)
            count += 1
        else:
            sequence.append(END_TOKEN)
            sequences.append(sequence)
            sequence = [token]
            count = 1
    if len(sequence) < seq_len - 1:
        diff = (seq_len - 1) - len(sequence)
        padding = [PADDING_TOKEN] * diff
        sequence.extend(padding)
    sequence.append(END_TOKEN)
    sequences.append(sequence)
    return sequences


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

class Dataset():
    def __init__(self, text_path, n = None, seq_len = None):
        self.text_path = text_path
        with open(text_path) as f:
            self.text = f.read()
        self.text = clean_text(self.text)
        self.tokens = tokenize(self.text)
        self.seqs = get_sequences(self.tokens, seq_len)
        self.counts = count_tokens(self.tokens)
        if n:
            self.counts = filter_n(self.counts, n)
        self.vocab_size = len(self.counts)
        self.sorted_tokens = sort_tokens_by_count(self.counts)