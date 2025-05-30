import numpy as np

def KF(xKF, P0, zMea, Phi, Gamma, H, Q, R, I, t):
    # 时间预测
    xKF_pre = np.dot(Phi, xKF[:, t - 1])
    P_pre = np.dot(np.dot(Phi, P0), Phi.T) + np.dot(np.dot(Gamma, Q), Gamma.T)
    # 测量更新
    K = np.dot(np.dot(P_pre, H.T), np.linalg.pinv(np.dot(np.dot(H, P_pre), H.T) + R))
    xFilter = xKF_pre + np.dot(K, (zMea[:, t] - np.dot(H, xKF_pre)))
    P0 = np.dot((I - np.dot(K, H)), P_pre)
    return xFilter, P0