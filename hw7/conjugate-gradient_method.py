import numpy as np

def conjugate_gradient(A, b, max_iter=1000, tol=1e-6):
    """
    共軛梯度法求解線性方程组 Ax = b（A 需為對稱正定矩陣）。
    
    參數:
        A: 係數矩陣 (n x n，對稱正定)
        b: 右側向量 (n x 1)
        max_iter: 最大迭代次數
        tol: 收斂容忍誤差
    
    返回:
        x: 解向量
        k: 實際迭代次數
    """
    n = len(b)
    x = np.zeros(n)  # 初始化解向量（全零）
    r = b - np.dot(A, x)  # 初始殘差
    p = r.copy()  # 初始搜索方向
    rsold = np.dot(r, r)  # 殘差的內積
    
    for k in range(max_iter):
        Ap = np.dot(A, p)  # 避免重複計算
        alpha = rsold / np.dot(p, Ap)  # 計算步長
        x += alpha * p  # 更新解
        r -= alpha * Ap  # 更新殘差
        rsnew = np.dot(r, r)  # 新殘差的內積
        
        # 檢查收斂條件
        if np.sqrt(rsnew) < tol:
            break
            
        # 更新搜索方向
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x, k + 1


# 正確定義的矩陣 A 和向量 b（A 需為對稱正定）
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

# 檢查 A 是否對稱
print("A 是否對稱:", np.allclose(A, A.T))

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)  # 修正後的右側項（第二方程為 -2）

# 呼叫共軛梯度法求解
x, iterations = conjugate_gradient(A, b)

# 輸出結果
print("解向量 x:", x)
print("迭代次數:", iterations)
print("驗證 Ax - b:", np.dot(A, x) - b)  # 計算殘差