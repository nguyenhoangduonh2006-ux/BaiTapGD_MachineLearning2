"""
Bài 3.27 - Thuật toán Perceptron
================================

Đề bài:
    Cho w = [1, 2, -10]^T, x = [3, 4, 1]^T (điểm dữ liệu đã thêm bias,
    phần tử cuối x0 = 1 ứng với bias).
    1. Tính w^T x.
    2. Xác định nhãn dự đoán của điểm dữ liệu.
    3. Nếu nhãn thực tế là y = -1, điểm dữ liệu có bị phân lớp sai không?

Mô tả thuật toán Perceptron:
    - Nhãn dự đoán:  y_hat = sign(w^T x)   (+1 nếu w^T x >= 0, ngược lại -1)
    - Điểm bị phân lớp sai khi y_hat != y, tức y * (w^T x) < 0.
    - Nếu sai, cập nhật:  w_new = w + y * x
      (kéo đường phân chia về phía phân lớp đúng cho điểm này).
    - Lặp lại cho đến khi không còn điểm nào bị phân lớp sai.

Cách chạy:
    python perceptron_3_27.py
"""

import numpy as np


def predict(w, x):
    """Trả về nhãn dự đoán sign(w^T x): +1 hoặc -1."""
    return 1 if np.dot(w, x) >= 0 else -1


def is_misclassified(w, x, y):
    """Điểm bị phân lớp sai khi nhãn dự đoán khác nhãn thực tế."""
    return predict(w, x) != y


def perceptron_update(w, x, y):
    """Bước cập nhật Perceptron: w_{t+1} = w_t + y * x."""
    return w + y * x


def main():
    # Dữ liệu đề bài (đã thêm bias)
    w = np.array([1, 2, -10])
    x = np.array([3, 4, 1])
    y = -1  # nhãn thực tế

    # Câu 1: tính w^T x
    z = np.dot(w, x)
    print("Câu 1: w^T x = 1*3 + 2*4 + (-10)*1 =", z)

    # Câu 2: nhãn dự đoán
    y_hat = predict(w, x)
    print("Câu 2: nhãn dự đoán y_hat = sign(%d) = %+d" % (z, y_hat))

    # Câu 3: kiểm tra phân lớp sai
    wrong = is_misclassified(w, x, y)
    print("Câu 3: nhãn thực tế y = %+d" % y)
    print("       y * w^T x = %d -> %s"
          % (y * z, "BỊ phân lớp sai" if wrong else "phân lớp đúng"))

    # Mở rộng: cập nhật w theo Perceptron nếu bị phân lớp sai
    if wrong:
        w_new = perceptron_update(w, x, y)
        z_new = np.dot(w_new, x)
        print("\nCập nhật: w_new = w + y*x =", w_new.tolist())
        print("Kiểm tra: w_new^T x =", z_new,
              "-> nhãn dự đoán mới =", "%+d" % predict(w_new, x))


if __name__ == "__main__":
    main()
