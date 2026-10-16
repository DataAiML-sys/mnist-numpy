import numpy as np

class NeuralNetwork:

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        exp_z = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.probs
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def backward(self, X, Y):
        batch_size = X.shape[0]
        dZ2 = self.probs - Y
        dW2 = self.a1.T @ dZ2 / batch_size
        db2 = np.sum(dZ2, axis=0) / batch_size
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.z1 > 0)
        dW1 = X.T @ dZ1 / batch_size
        db1 = np.sum(dZ1, axis=0) / batch_size
        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2