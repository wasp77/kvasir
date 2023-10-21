import numpy as np
import math
from embedding import EmbeddingLayer


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def normalize(X, gamma=1, beta=0, epsilon=1e-6):
    mean = np.mean(X, axis=-1, keepdims=True)
    std_dev = np.std(X, axis=-1, keepdims=True)
    return gamma * (X - mean) / (std_dev + epsilon) + beta


class NormLayer():
    def __init__(self, gamma=1, beta=0, epsilon=1e-6):
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, X):
        self.X = X
        self.mean = np.mean(X, axis=-1, keepdims=True)
        self.std_dev = np.std(X, axis=-1, keepdims=True)
        return self.gamma * (X - self.mean) / (self.std_dev + self.epsilon) + self.beta

    def backwards(self, grad):
        N = grad.shape[0]
        dnorm = grad * self.gamma
        dvar = np.sum(dnorm * (self.X - self.mean) * -0.5 *
                      np.power(self.std_dev + self.epsilon, -1.5), axis=1, keepdims=True)
        dmean = np.sum(dnorm * -1 / self.std_dev, axis=1, keepdims=True) + \
            dvar * np.mean(-2*(self.X - self.mean), axis=1, keepdims=True)
        dx = (dnorm / self.std_dev) + (dvar * 2 *
                                       (self.X - self.mean) / N) + (dmean / N)
        return dx


class AttentionHead():
    def __init__(self, lr=.01, dk=64, d_model=128):
        self.lr = lr
        self.dk = dk
        self.d_model = d_model

        self.Wq = np.random.rand(d_model, dk)
        self.Wk = np.random.rand(d_model, dk)
        self.Wv = np.random.rand(d_model, dk)

        self.temp_q_grad = None
        self.temp_k_grad = None
        self.temp_v_grad = None

    def forward(self, X):
        self.X = X
        self.Q = np.dot(X, self.Wq)
        self.K = np.dot(X, self.Wk)
        self.V = np.dot(X, self.Wv)

        self.scores = np.dot(self.Q, self.K.T)
        self.scaled_scores = self.scores / math.sqrt(self.dk)
        self.attention_weights = softmax(self.scaled_scores)
        return np.dot(self.attention_weights, self.V)

    def backwards(self, grad):
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
        dWv = np.dot(self.X.T, dV.T)

        self.temp_q_grad = dWq
        self.temp_k_grad = dWk
        self.temp_v_grad = dWv

        # self.Wq -= self.lr * dWq
        # self.Wk -= self.lr * dWk
        # self.Wv -= self.lr * dWv

        dX_q = np.dot(dQ, self.Wq.T)
        dX_k = np.dot(dK, self.Wk.T)
        dX_v = np.dot(dV.T, self.Wv.T)

        return dX_q + dX_k + dX_v

    def update_weights(self, scaling_factor):
        self.temp_q_grad *= scaling_factor
        self.temp_k_grad *= scaling_factor
        self.temp_v_grad *= scaling_factor

        self.Wq -= self.lr * self.temp_q_grad
        self.Wk -= self.lr * self.temp_k_grad
        self.Wv -= self.lr * self.temp_v_grad

    def get_grads(self):
        return np.concatenate([self.temp_k_grad.ravel(), self.temp_q_grad.ravel(), self.temp_v_grad.ravel()])


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
        # He initialization.
        self.W = np.random.randn(rows, columns) * np.sqrt(2. / rows)
        self.activation = activation
        self.temp_grad = None

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
        self.temp_grad = dW
        return np.dot(self.W, dz.T).T

    def update_weights(self, scaling_factor):
        self.temp_grad *= scaling_factor
        self.W -= self.lr * self.temp_grad


relu_activation = ActivationFunc(
    func=relu, grad_func=lambda x: np.piecewise(x, [x <= 0, x > 0], [0, 1]))


class FeedForwardNetwork():
    def __init__(self, input_size, output_size, vocab_size, lr=.01):
        self.input_layer = LinearLayer(
            rows=input_size, columns=output_size, lr=lr)
        self.hidden_layer = LinearLayer(
            rows=output_size, columns=output_size, activation=relu_activation, lr=lr)
        self.output_layer = LinearLayer(
            rows=output_size, columns=vocab_size, lr=lr)

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

    def get_grads(self):
        return np.concatenate([self.input_layer.temp_grad.ravel(), self.hidden_layer.temp_grad.ravel(), self.output_layer.temp_grad.ravel()])

    def update_weights(self, scaling_factor):
        self.input_layer.update_weights(scaling_factor=scaling_factor)
        self.hidden_layer.update_weights(scaling_factor=scaling_factor)
        self.output_layer.update_weights(scaling_factor=scaling_factor)


class CategoricalCrossEntropyLoss():
    def calc_loss(self, output, y_true):
        self.y_pred = softmax(output)
        self.y_true = y_true
        self.loss = -np.sum(self.y_true * np.log(self.y_pred + 1e-9))
        return self.loss

    def get_grad(self):
        return self.y_pred - self.y_true


def gen_gradient_clip_scaler(grad, threshold=1.0):
    square = np.square(grad)
    sum = np.sum(square)
    l2 = np.sqrt(sum)
    if l2 > threshold:
        return threshold / l2
    return 1.0


class SimpleTransformer():
    def __init__(self, vocab_size, tokens, seq_len, d_model=128, dk=64, lr=0.01):
        self.d_model = d_model
        self.embedding_layer = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model, seq_len=seq_len, lr=lr)
        self.feed_forward = FeedForwardNetwork(
            input_size=dk, output_size=d_model, vocab_size=vocab_size, lr=lr)
        self.attention_head = AttentionHead(d_model=d_model, lr=lr, dk=dk)
        self.embedding_layer.gen_token_mapping(tokens)
        self.norm_layer1 = NormLayer()
        self.norm_layer2 = NormLayer()

    def forward(self, seq):
        self.tokens = seq
        embeddings = self.embedding_layer.forward(self.tokens)
        X = self.attention_head.forward(embeddings)
        X = self.norm_layer1.forward(X)
        X = self.feed_forward.forward(X)
        X = self.norm_layer2.forward(X)
        return X

    def backwards(self, grad):
        grad = self.norm_layer2.backwards(grad)
        grad = self.feed_forward.backwards(grad)
        grad = self.norm_layer1.backwards(grad)
        grad = self.attention_head.backwards(grad)

        grads = np.concatenate([grad.ravel(), self.feed_forward.get_grads(
        ).ravel(), self.attention_head.get_grads().ravel()])
        scaling_factor = gen_gradient_clip_scaler(grads)
        self.feed_forward.update_weights(scaling_factor)
        self.attention_head.update_weights(scaling_factor)
        grad *= scaling_factor

        self.embedding_layer.backwards(grad, self.tokens)

    def get_token_mapping(self):
        return self.embedding_layer.token_to_id
