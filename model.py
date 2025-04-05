import numpy as np
np.random.seed(42)

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.params['b2'] = np.zeros(output_size)
        self.activation = activation

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 前向传播
        self.z1 = X.dot(W1) + b1
        if self.activation == 'relu':
            self.a1 = np.maximum(0, self.z1)
        elif self.activation == 'sigmoid':
            self.a1 = 1 / (1 + np.exp(-self.z1))
        elif self.activation == 'tanh':
            self.a1 = np.tanh(self.z1)

        self.scores = self.a1.dot(W2) + b2
        exp_scores = np.exp(self.scores - np.max(self.scores, axis=1, keepdims=True))
        self.probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, reg):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        num_samples = X.shape[0]

        # 计算梯度
        dscores = self.probs
        dscores[range(num_samples), y] -= 1
        dscores /= num_samples

        dW2 = self.a1.T.dot(dscores)
        db2 = np.sum(dscores, axis=0)
        da1 = dscores.dot(W2.T)

        if self.activation == 'relu':
            dz1 = da1 * (self.z1 > 0)
        elif self.activation == 'sigmoid':
            dz1 = da1 * (self.a1 * (1 - self.a1))
        elif self.activation == 'tanh':
            dz1 = da1 * (1 - self.a1 ** 2)

        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0)

        # 添加正则化项梯度
        dW2 += reg * W2
        dW1 += reg * W1

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return grads