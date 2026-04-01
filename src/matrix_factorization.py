import numpy as np

class MatrixFactorization:
    def __init__(self, R, K=20, lr=0.01, reg=0.1, epochs=10):
        self.R = R.fillna(0).values
        self.num_users, self.num_items = self.R.shape
        self.K = K
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

    def fit(self):
        self.U = np.random.rand(self.num_users, self.K)
        self.V = np.random.rand(self.num_items, self.K)

        rows, cols = np.nonzero(self.R)

        for _ in range(self.epochs):
            for i, j in zip(rows, cols):
                error = self.R[i, j] - np.dot(self.U[i], self.V[j])
                self.U[i] += self.lr*(error*self.V[j] - self.reg*self.U[i])
                self.V[j] += self.lr*(error*self.U[i] - self.reg*self.V[j])

    def predict(self):
        return np.dot(self.U, self.V.T)