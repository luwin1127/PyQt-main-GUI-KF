import numpy as np

def udu(P):
    # P = U * diag(D) * U.T, U为上三角矩阵
    n = len(P)
    U = np.eye(n)
    D = np.zeros((n, 1))
    trPn = np.trace(P) / n * 1e-40
    for j in range(n - 1, -1, -1):
        k = slice(j + 1, n)
        D[j] = P[j, j] - np.dot(U[j, k] ** 2, D[k])
        if D[j] <= trPn:
            continue
        for i in range(j - 1, -1, -1):
            U[i, j] = (P[i, j] - np.dot(U[i, k] * U[j, k], D[k])) / D[j]
    return U, D