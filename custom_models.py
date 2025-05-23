
import numpy as np

class LogisticRegressionCustom:
    def __init__(self, input_dim, lr=0.01, epochs=100):
        self.weights = np.random.randn(input_dim)
        self.bias = 0
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        for _ in range(self.epochs):
            predictions = self.sigmoid(X @ self.weights + self.bias)
            error = predictions - y
            grad_weights = X.T @ error / len(X)
            grad_bias = np.mean(error)
            self.weights -= self.lr * grad_weights
            self.bias -= self.lr * grad_bias

    def predict(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def accuracy(self, X, y):
        preds = (self.predict(X) > 0.5).astype(int)
        return np.mean(preds == y)
