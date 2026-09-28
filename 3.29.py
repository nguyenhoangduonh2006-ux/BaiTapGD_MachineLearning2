"""
Bài 3.29 - Lớp Perceptron đơn giản
- fit(X, y): huấn luyện, nhãn y là +1 hoặc -1
- predict(X): dự báo nhãn cho dữ liệu mới
"""
import numpy as np


class Perceptron:
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.w = None

    def fit(self, X, y):
        X = np.hstack((np.ones((len(X), 1)), X))   # thêm cột bias
        self.w = np.zeros(X.shape[1])              # w ban đầu = 0

        for _ in range(self.max_iter):
            co_loi = False
            for xi, yi in zip(X, y):
                if yi * np.dot(self.w, xi) <= 0:   # điểm bị phân lớp sai
                    self.w = self.w + yi * xi      # cập nhật w
                    co_loi = True
            if not co_loi:                         # không còn điểm sai -> dừng
                break
        return self

    def predict(self, X):
        X = np.hstack((np.ones((len(X), 1)), X))
        return np.where(X @ self.w >= 0, 1, -1)


if __name__ == "__main__":
    X = np.array([[2, 2], [3, 3], [2, 3], [-2, -2], [-3, -1], [-2, -3]])
    y = np.array([1, 1, 1, -1, -1, -1])

    model = Perceptron()
    model.fit(X, y)
    print("w =", model.w)
    print("Dự báo tập huấn luyện:", model.predict(X))
    print("Dự báo điểm mới [[4, 4], [-4, -2]]:", model.predict(np.array([[4, 4], [-4, -2]])))
