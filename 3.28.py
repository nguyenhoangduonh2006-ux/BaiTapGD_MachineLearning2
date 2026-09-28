"""
Bài 3.28 - Thuật toán Perceptron
================================

Đề bài:
    Cho w = [-2, 1, 0]^T, x = [2, 3, 1]^T, y = 1
    (x đã thêm bias, phần tử cuối x0 = 1 ứng với bias).
    1. Kiểm tra mẫu có bị phân lớp sai hay không.
    2. Nếu sai, thực hiện một bước cập nhật Perceptron.
    3. Tính lại giá trị w^T x sau cập nhật.

Mô tả thuật toán Perceptron:
    - Nhãn dự đoán:  y_hat = sign(w^T x)   (+1 nếu w^T x >= 0, ngược lại -1)
    - Điểm bị phân lớp sai khi y_hat != y, tức y * (w^T x) < 0.
    - Nếu sai, cập nhật:  w_new = w + y * x
    - Sau cập nhật, w^T x thay đổi theo y * ||x||^2, nghĩa là được đẩy
      về phía nhãn đúng của điểm dữ liệu.

Cách chạy:
    python perceptron_3_28.py
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
    w = np.array([-2, 1, 0])
    x = np.array([2, 3, 1])
    y = 1  # nhãn thực tế

    # Câu 1: kiểm tra phân lớp sai
    z = np.dot(w, x)
    y_hat = predict(w, x)
    wrong = is_misclassified(w, x, y)
    print("Câu 1: w^T x = (-2)*2 + 1*3 + 0*1 =", z)
    print("       nhãn dự đoán y_hat = %+d, nhãn thực tế y = %+d" % (y_hat, y))
    print("       y * w^T x = %d -> %s"
          % (y * z, "BỊ phân lớp sai" if wrong else "phân lớp đúng"))

    # Câu 2: cập nhật Perceptron nếu sai
    if wrong:
        w_new = perceptron_update(w, x, y)
        print("\nCâu 2: w_new = w + y*x =", w_new.tolist())

        # Câu 3: tính lại w^T x
        z_new = np.dot(w_new, x)
        print("Câu 3: w_new^T x = 0*2 + 4*3 + 1*1 =", z_new)
        print("       nhãn dự đoán mới = %+d -> %s"
              % (predict(w_new, x),
                 "phân lớp đúng" if not is_misclassified(w_new, x, y)
                 else "vẫn sai"))
    else:
        print("\nMẫu đã được phân lớp đúng, không cần cập nhật.")


if __name__ == "__main__":
    main()
