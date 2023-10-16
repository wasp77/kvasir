import numpy as np
import math
from embedding import EmbeddingLayer

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


class ActivationFunc():
    def __init__(self, func, grad_func):
        self.func = func
        self.grad_func = grad_func

    def __call__(self, *args):
        self.result = self.func(*args)
        return self.result

    def grad(self):
        return self.grad_func(self.result)


class LinearLayer():
    def __init__(self, rows, columns, lr=.01, activation=None):
        self.rows = rows
        self.columns = columns
        self.lr = lr
        self.W = np.random.rand(rows, columns)
        self.activation = activation

    def forward(self, X):
        self.X = X
        if self.activation:
            return self.activation(np.dot(X, self.W))
        return np.dot(X, self.W)

    def backwards(self, grad):
        if self.activation:
            da = self.activation.grad()
            dz = grad * da
        else:
            dz = grad

        dW = np.dot(self.X.T, dz)
        self.W -= self.lr * dW
        return np.dot(self.W, dz)


relu_activation = ActivationFunc(
    func=relu, grad_func=lambda x: np.piecewise(x, [x <= 0, x > 0], [0, 1]))


class FeedForwardNetwork():
    def __init__(self, vocab_size, d_model, lr=.01):
        self.input_layer = LinearLayer(rows=vocab_size, columns=d_model, lr=lr)
        self.hidden_layer = LinearLayer(
            rows=d_model, columns=d_model, activation=relu_activation, lr=lr)
        self.output_layer = LinearLayer(
            rows=d_model, columns=vocab_size, lr=lr)

    def forward(self, X):
        X = self.input_layer.forward(X)
        X = self.hidden_layer.forward(X)
        X = self.output_layer.forward(X)
        return X

    def backwards(self, grad):
        grad = self.output_layer.backwards(grad)
        grad = self.hidden_layer.backwards(grad)
        grad = self.input_layer.backwards(grad)
        return grad
