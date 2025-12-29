import numpy as np

class SimpleCNN:
    def __init__(self):
        self.filter = np.random.randn(3, 3)
        self.W = np.random.randn(4, 2)

    def conv2d(self, image):
        h, w = image.shape
        output = np.zeros((h - 2, w - 2))
        for i in range(h - 2):
            for j in range(w - 2):
                output[i, j] = np.sum(image[i:i+3, j:j+3] * self.filter)
        return output

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, image):
        conv = self.relu(self.conv2d(image))
        flat = conv.flatten()
        out = flat @ self.W
        return out
