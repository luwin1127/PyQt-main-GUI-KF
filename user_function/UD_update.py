import numpy as np

def UD_update(U, D, Phi, Gamma, Q, H, R, TM):
    n = len(U)
    if 'T' in TM.upper():  # Thornton算法的UD时间更新
        W = np.hstack((np.dot(Phi, U), Gamma))
        D1 = np.vstack((D, Q))
        for j in range(n - 1, -1, -1):
            D[j] = np.dot(W[j, :] ** 2, D1)
            for i in range(j):
                U[i, j] = np.dot(W[i, :] * W[j, :], D1) / D[j]
                W[i, :] = W[i, :] - U[i, j] * W[j, :]
    if 'M' in TM.upper():  # Bierman算法的UD测量更新
        f = np.dot(H, U).T
        g = D * f
        afa = np.dot(f.T, g) + R
        for j in range(n - 1, -1, -1):
            afa0 = afa - f[j] * g[j]
            lambda_ = -f[j] / afa0
            D[j] = (afa0 / afa) * D[j]
            afa = afa0
            for i in range(j - 1, -1, -1):
                s = slice(i + 1, j)
                U[i, j] = U[i, j] + lambda_ * (g[i] + np.dot(U[i, s], g[s]))
    return U, D