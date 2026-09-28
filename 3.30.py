"""
Bài 3.30 - Phân lớp nhị phân bằng Perceptron
Dữ liệu : Breast Cancer (sklearn) - chẩn đoán khối u ác tính (1) hay lành tính (0)
Các bước: chia train/test -> chuẩn hóa -> tìm tham số tốt nhất (GridSearchCV theo F1)
          -> đánh giá bằng Accuracy, Precision, Recall, F1-score
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Tải dữ liệu và chia train/test
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Chuẩn hóa dữ liệu (Perceptron rất nhạy với thang đo)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Mô hình gốc (tham số mặc định) để so sánh
base = Perceptron(random_state=42).fit(X_train, y_train)
print("Mô hình mặc định:")
pred = base.predict(X_test)
print("  Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f" % (
    accuracy_score(y_test, pred), precision_score(y_test, pred),
    recall_score(y_test, pred), f1_score(y_test, pred)))

# 4. Tìm tham số tốt nhất theo F1-score (cross-validation 5 fold)
params = {
    "alpha": [0.0001, 0.001, 0.01, 0.1],   # hệ số điều chuẩn
    "penalty": [None, "l2", "l1"],
    "eta0": [0.1, 0.5, 1.0],               # tốc độ học
    "max_iter": [100, 500, 1000],
}
grid = GridSearchCV(Perceptron(random_state=42), params, scoring="f1", cv=5)
grid.fit(X_train, y_train)
print("\nTham số tốt nhất:", grid.best_params_)

# 5. Đánh giá mô hình tốt nhất trên tập test
best = grid.best_estimator_
pred = best.predict(X_test)
print("\nMô hình sau khi tối ưu:")
print("  Accuracy  = %.4f" % accuracy_score(y_test, pred))
print("  Precision = %.4f" % precision_score(y_test, pred))
print("  Recall    = %.4f" % recall_score(y_test, pred))
print("  F1-score  = %.4f" % f1_score(y_test, pred))
