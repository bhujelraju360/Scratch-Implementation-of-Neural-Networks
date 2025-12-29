import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.h = np.zeros((1, hidden_size))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, inputs):
        for x in inputs:
            x = x.reshape(1, -1)
            self.h = self.tanh(x @ self.Wx + self.h @ self.Wh)
        y = self.h @ self.Wy
        return y
