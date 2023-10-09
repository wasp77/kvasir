import numpy as np
import math

dk = 64
d_model = 128

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def normalize(X, gamma=1, beta=0, epsilon=1e-6):
    mean = np.mean(X, axis=-1, keepdims=True)
    std_dev = np.std(X, axis=-1, keepdims=True)
    return gamma * (X - mean) / (std_dev + epsilon) + beta

class AttentionHead():
    def __init__(self, lr=.01, dk=64, d_model=128):
        self.lr = lr
        self.dk = dk
        self.d_model = d_model

        self.Wq = np.random.rand(d_model, dk)
        self.Wk = np.random.rand(d_model, dk)
        self.Wv = np.random.rand(d_model, dk)

    def forward(self, X):
        self.X = X
        self.Q = np.dot(X, self.Wq)
        self.K = np.dot(X, self.Wk)
        self.V = np.dot(X, self.Wv)

        self.scores = np.dot(self.Q, self.K.T)
        self.scaled_scores = self.scores / math.sqrt(self.dk) 
        self.attention_weights = softmax(self.scaled_scores)
        return np.dot(self.attention_weights, self.V)

    def backward(self, grad):
        # gradient attention_weights
        da = np.dot(grad, self.V.T)

        # gradient V
        dV = np.dot(grad.T, self.attention_weights) 

        # gradient scaling scores
        sum_term = np.sum(self.attention_weights * da, axis=1)
        dss = self.attention_weights * (da - sum_term[:, np.newaxis])

        # gradient scores
        ds = dss * np.sqrt(self.dk)

        # gradient Q
        dQ = np.dot(ds, self.K)

        # gradient K
        dK = np.dot(ds.T, self.Q)

        # weight grandients
        dWq = np.dot(self.X.T, dQ)
        dWk = np.dot(self.X.T, dK)
        dWv = np.dot(self.X.T, dV)

        self.Wq -= self.lr * dWq
        self.Wk -= self.lr * dWk
        self.Wv -= self.lr * dWv


dff = 256
W1 = np.random.rand(d_model, dff)
b1 = np.random.rand(dff)
W2 = np.random.rand(dff, d_model)
b2 = np.random.rand(d_model)


def relu(x):
    return np.maximum(x, 0)

class LinearLayer():
    def __init__(self, rows, columns, lr=.01):
        self.rows = rows
        self.columns = columns
        self.lr = lr
        self.W = np.random.rand(rows, columns)

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W)

    def backwards(self, grad):
        dW = np.dot(grad, self.X.T)
        self.W -= self.lr * dW
        return np.dot(grad, self.W.T)
