import numpy as np


# 1. Hàm tính giá trị hàm số (cost function)
def cost(x):
    return x**2 - 4*x + 5


# 2. Hàm tính đạo hàm (gradient function)
def grad(x):
    return 2*x - 4


# 3. Thuật toán Gradient Descent (myGD1)
def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:  # Điều kiện dừng khi đạo hàm đủ nhỏ
            break
        x.append(x_new)
    return (x, it)


if __name__ == "__main__":
    # --- THỰC THI CHƯƠNG TRÌNH ---
    x0 = 5
    eta = 0.2

    # Lấy lịch sử cập nhật qua các bước
    (x_hist, total_its) = myGD1(x0, eta)

    print("=== CHI TIẾT 4 BƯỚC ĐẦU TIÊN ===")
    for t in range(min(5, len(x_hist))):
        x_t = x_hist[t]
        c_t = cost(x_t)
        g_t = grad(x_t)
        print(f"Bước {t}: x_{t} = {x_t:.4f} | f(x_{t}) = {c_t:.6f} | f'(x_{t}) = {g_t:.4f}")

    print("\n=== KẾT QUẢ HỘI TỤ CHUNG ===")
    print('Solution x = %f, cost = %f, after %d iterations'
          % (x_hist[-1], cost(x_hist[-1]), total_its))
