import numpy as np

def sor(A, b, omega=1.2, max_iter=1000, tol=1e-6):
    """
    SOR 迭代法求解線性方程组 Ax = b。
    
    參數:
        A: 係數矩陣 (n x n)
        b: 右側向量 (n x 1)
        omega: 鬆弛參數 (1 < omega < 2 時加速收斂)
        max_iter: 最大迭代次數
        tol: 收斂容忍誤差
    
    返回:
        x: 解向量
        iterations: 實際迭代次數
    """
    n = len(b)
    x = np.zeros(n)  # 初始化解向量（全零）
    
    for k in range(max_iter):
        x_old = x.copy()  # 保存當前解向量，用於收斂判斷
        
        for i in range(n):
            # 計算 sigma = sum(a_ij * x_j) 其中 j != i
            # 分為兩部分：j < i 用新值，j > i 用舊值
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sigma)
        
        # 檢查收斂條件：如果新舊解的差異小於容忍誤差，則停止迭代
        if np.linalg.norm(x - x_old) < tol:
            break
    
    return x, k + 1


# 正確定義的矩陣 A 和向量 b（根據題目要求）
A = np.array([
    [4, -1, 0, -1, 0, 0],    # 方程 (1): 4x1 -x2 -x4 = 0
    [-1, 4, -1, 0, -1, 0],    # 方程 (2): -x1 +4x2 -x3 -x5 = -1
    [0, -1, 4, 0, 1, -1],     # 方程 (3): -x2 +4x3 +x5 -x6 = 9
    [-1, 0, 0, 4, -1, -1],    # 方程 (4): -x1 +4x4 -x5 -x6 = 4
    [0, -1, 0, -1, 4, -1],    # 方程 (5): -x2 -x4 +4x5 -x6 = 8
    [0, 0, -1, 0, -1, 4]      # 方程 (6): -x3 -x5 +4x6 = 6
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)  # 方程右側常數項

# 呼叫 SOR 方法求解（omega=1.2 為經驗值，可調整）
x, iterations = sor(A, b, omega=1.2)

# 輸出結果
print("解向量 x:", x)
print("迭代次數:", iterations)
print("驗證 Ax - b:", np.dot(A, x) - b)  # 計算殘差