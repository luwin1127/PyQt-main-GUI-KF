import sys
import pickle
import numpy as np
from datetime import datetime
from user_function.KF import KF
from user_function.UD_update import UD_update
from user_function.udu import udu
from user_function.filter_plot import filter_plot
from PyQt5 import QtCore, QtGui, QtWidgets
import Ui_mainGUI_alpha

def main():
    # 选择滤波模式
    mode_flag = 1
    mode_options = {
        1: 'null',
        2: 'Problem 1',
        3: 'Problem 2',
        4: 'Problem 3',
        5: 'Problem 4-1',
        6: 'Problem 4-2'
    }
    mode = mode_options[mode_flag]

    # 保存数据参数
    save_fig = True                     # 是否保存图片
    save_data = True                    # 是否保存数据
    with_time_flag = False               # 保存数据是否带日期

    # 仿真参数
    T = 300                             # 仿真时长
    Ts = np.random.uniform(0.1, 1.0)    # 时间更新间隔(0.1-1.0之间随机小数)
    Q = np.random.uniform(0.01, 0.1) * Ts  # 过程噪声
    R = np.random.uniform(0.1, 1.0)     # 量测噪声

    # 生成噪声序列
    W = np.sqrt(Q) * np.random.randn(1, T)
    V = np.sqrt(R) * np.random.randn(1, T)
    p0 = np.random.uniform(1, 10, 2)     # 初始协方差阵参数
    P0 = np.diag([p0[0], p0[1]])         # 初始协方差阵
    I0 = np.array([[0., 0.], [0., 0.]])  # 初始信息阵设置为0

    # 系统矩阵
    A = np.array([[0, 1], [0, 0]])        # 状态矩阵
    I = np.eye(2)                         # 单位阵
    Phi = I + A * Ts                      # 离散化
    H = np.array([[1, 0]])                # 量测矩阵
    Gamma = np.array([[0], [1]])

    # 设置滤波维度
    nS = 2                                # 状态维度
    nZ = 1                                # 观测维度

    # 分配空间
    x_state = np.zeros((nS, T))            # 系统真实值
    z_mea = np.zeros((nZ, T))              # 系统观测值
    x_KF = np.zeros((nS, T))               # 卡尔曼滤波状态值

    # 赋初值
    x_state[:, 0] = np.random.uniform(-10, 10, nS)  # 系统状态初值
    z_mea[:, 0] = np.dot(H, x_state[:, 0])                  # 系统观测初值(@ 代表矩阵乘法)
    x_KF[:, 0] = x_state[:, 0]                       # 卡尔曼滤波器估计初值

    # 02 用模型模拟真实状态
    for t in range(1, T):
        x_state[:, t] = np.dot(Phi,x_state[:, t-1]) + np.dot(Gamma, W[0, t]).squeeze()
        z_mea[:, t] = np.dot(H, x_state[:, t]) + V[0, t]

    # 03-1 Kalman滤波
    if mode == 'null':
        mode_str = 'KF'
        P0_kf = P0.copy()  # 复制初始P0
        for t in range(1, T):
            x_KF[:, t], P0_kf = KF(x_KF, P0_kf, z_mea, Phi, Gamma, H, Q, R, I, t)
    
        # 画图
        filter_plot(x_state, z_mea, x_KF, None, None, None, None, T, "KF", True, "KF")

    # 04-1 习题1：信息滤波
    if mode == 'Problem 1':
        mode_str = 'IF'
        # 分配空间并赋予滤波器变量初值
        s_IF = np.zeros((nS, T))  # 信息滤波状态值
        x_IF = np.zeros((nS, T))
        s_IF[:, 0] = np.dot(I0, x_state[:, 0])  # 信息滤波器赋估计初值
        Phi_inv = np.linalg.pinv(Phi)  # 先求逆，减少计算量
        Q_inv = 1/Q
        R_inv = 1/R

        # 重新初始化I0为协方差阵的逆
        I0 = np.linalg.inv(P0.copy())
        
        for t in range(1, T):
            M_k_1 = np.dot(np.dot(Phi_inv.T, I0), Phi_inv)
            N_k_1 = np.dot(np.dot(np.dot(M_k_1, Gamma), np.linalg.pinv(np.dot(np.dot(Gamma.T, M_k_1), Gamma) + Q_inv)), Gamma.T)
            S_pre = np.dot((I - N_k_1), np.dot(Phi_inv.T, s_IF[:, t-1]))
            I_pre = np.dot((I - N_k_1), M_k_1)
            s_IF[:, t] = S_pre + np.dot(H.T, np.dot(R_inv, z_mea[:, t]))
            I0 = I_pre + np.dot(H.T, np.dot(R_inv, H))
            x_IF[:, t] = np.dot(np.linalg.pinv(I0), s_IF[:, t])
        
        # 画图
        filter_plot(x_state, z_mea, x_KF, x_IF, None, None, None, T, mode, save_fig, mode_str)

    # 04-2 习题2：UD滤波
    elif mode == 'Problem 2':
        mode_str = 'UD'
        # 分配空间并赋予滤波器变量初值
        x_UD = np.zeros((nS, T))  # UD滤波状态值
        x_UD[:, 0] = x_state[:, 0]  # UD滤波器赋估计初值
        P0_ud = np.diag([10**2, 1**2])  # 初始协方差阵
        
        for t in range(1, T):
            # 时间更新
            xUD_pre = np.dot(Phi, x_UD[:, t-1])
            U, D = udu(P0_ud)  # 将协方差阵作UD分解
            U, D = UD_update(U, D, Phi, Gamma, Q, H, R, 'T')  # 时间更新
            P_pre = np.dot(np.dot(U, np.diag(D.flatten())), U.T)
            
            # 量测更新
            K = np.dot(np.dot(P_pre, H.T), np.linalg.pinv(np.dot(np.dot(H, P_pre), H.T) + R))
            x_UD[:, t] = xUD_pre + np.dot(K, (z_mea[:, t] - np.dot(H, xUD_pre)))
            U, D = udu(P_pre)  # 将一步预测协方差阵作UD分解
            U, D = UD_update(U, D, Phi, Gamma, Q, H, R, 'M')  # 测量更新
            P0_ud = np.dot(np.dot(U, np.diag(D.flatten())), U.T)
        
        # 画图
        filter_plot(x_state, z_mea, x_KF, None, x_UD, None, None, T, mode, save_fig, mode_str)

    # 04-3 习题3：遗忘滤波
    elif mode == 'Problem 3':
        mode_str = 'FD'
        # 根据题设条件，设置初始化参数
        a = 1  # 加速度(m/s^2)，减速机动，因为速度为负，所以这里符号为正
        acc = np.dot(np.array([0.5*Ts**2, Ts]), a)  # 构造矩阵
        t_100 = 100  # 时间(s)，导弹在第100秒时，减速机动
        t_200 = 200  # 时间(s)，导弹在第200秒后，停止减速机动，进入匀速运动
        
        # 根据题设条件，用模型模拟真实状态
        for t in range(1, T):
            if t < t_100 or t >= t_200:
                x_state[:, t] = np.dot(Phi, x_state[:, t-1]) + np.dot(Gamma, W[0, t]).squeeze()
            else:
                x_state[:, t] = np.dot(Phi, x_state[:, t-1]) + acc + np.dot(Gamma, W[0, t]).squeeze()
            z_mea[:, t] = np.dot(H, x_state[:, t]) + V[0, t]
        
        # 分配空间并赋予滤波器变量初值
        s = 1.1  # 渐消因子
        x_KF = np.zeros((nS, T))  # 卡尔曼滤波状态值
        x_KF[:, 0] = x_state[:, 0]  # 卡尔曼滤波器估计初值
        x_FD = np.zeros((nS, T))  # 遗忘滤波状态值
        x_FD[:, 0] = x_state[:, 0]  # 遗忘滤波器赋估计初值
        
        # 再做一次卡尔曼滤波
        P0_kf = np.diag([10**2, 1**2])  # 重新设置初始P0
        for t in range(1, T):
            x_KF[:, t], P0_kf = KF(x_KF, P0_kf, z_mea, Phi, Gamma, H, Q, R, I, t)
        
        # 遗忘滤波
        P0_fd = np.diag([10**2, 1**2])  # 重新设置初始P0
        for t in range(1, T):
            # 时间更新
            xFD_pre = np.dot(Phi, x_FD[:, t-1])
            P_pre = np.dot(np.dot(Phi, s * P0_fd), Phi.T) + np.dot(np.dot(Gamma, Q), Gamma.T)
            
            # 量测更新
            K = np.dot(np.dot(P_pre, H.T), np.linalg.pinv(np.dot(np.dot(H, P_pre), H.T) + R))
            x_FD[:, t] = xFD_pre + np.dot(K, (z_mea[:, t] - np.dot(H, xFD_pre)))
            P0_fd = np.dot((I - np.dot(K, H)), P_pre)
        
        # 画图
        filter_plot(x_state, z_mea, x_KF, None, None, x_FD, None, T, mode, save_fig, mode_str)

    # 04-4-1 习题4（1）
    elif mode == 'Problem 4-1':
        mode_str = 'FT1'
        R_1s = 100  # 量测噪声方差为100m^2
        R_100s = 10000  # 量测噪声方差为10000m^2
        
        # 构造量测噪声序列
        V = np.zeros((1, T))
        V[0, :100] = np.sqrt(R_1s) * np.random.randn(100)  # 1-100s，方差为100m^2
        V[0, 100:] = np.sqrt(R_100s) * np.random.randn(T-100)  # 100s之后，方差为10000m^2
        
        # 根据题设条件，用模型模拟真实状态
        for t in range(1, T):
            if t == 150:  # 在150s时出现量测故障导致雷达输出为0
                z_mea[:, t] = 0
            else:
                z_mea[:, t] = np.dot(H, x_state[:, t]) + V[0, t]
        
        # 再做一次卡尔曼滤波
        P0_kf = np.diag([10**2, 1**2])  # 重新设置初始P0
        for t in range(1, T):
            # 时间更新
            xKF_pre = np.dot(Phi, x_KF[:, t-1])
            P_pre = np.dot(np.dot(Phi, P0_kf), Phi.T) + np.dot(np.dot(Gamma, Q), Gamma.T)
            
            # 量测更新
            K = np.dot(np.dot(P_pre, H.T), np.linalg.pinv(np.dot(np.dot(H, P_pre), H.T) + R_1s))
            x_KF[:, t] = xKF_pre + np.dot(K, (z_mea[:, t] - np.dot(H, xKF_pre)))
            P0_kf = np.dot((I - np.dot(K, H)), P_pre)
        
        # 分配空间并赋予滤波器变量初值
        x_fault = np.zeros((nS, T))  # 自适应遗忘滤波状态值
        x_fault[:, 0] = x_state[:, 0]  # 自适应遗忘滤波器赋估计初值
        beta = np.zeros(T)
        beta[0] = 1
        b = 0.999  # 渐消因子
        C = np.dot(np.dot(H, (np.dot(np.dot(Phi, P0), Phi.T) + np.dot(np.dot(Gamma, Q), Gamma.T))), H.T) + R  # 新息序列方差阵
        zNew = 15
        
        # 量测噪声方差自适应滤波处理
        for t in range(1, T):
            # 时间更新
            xFault_pre = np.dot(Phi, x_fault[:, t-1])  # 状态一步预测
            P_pre = np.dot(np.dot(Phi, P0), Phi.T) + np.dot(np.dot(Gamma, Q), Gamma.T)  # 协方差一步预测
            
            # 量测更新
            beta[t] = beta[t-1] / (beta[t-1] + b)  # 更新新息序列方差阵的系数
            C = (1 - beta[t]) * C + beta[t] * (zNew * zNew)
            
            if abs(np.trace(C) / np.trace(np.dot(np.dot(H, P_pre), H.T) + R_1s)) > 2:  # 量测故障检测与隔离
                x_fault[:, t] = xFault_pre  # 无需量测更新，估计值用时间更新值代替
                P0 = P_pre
            else:
                alpha = np.trace(C - np.dot(np.dot(H, P_pre), H.T)) / np.trace(np.array([[R_1s]])) # 将 R_1s 转换为 1x1 矩阵
                K = np.dot(np.dot(P_pre, H.T), np.linalg.pinv(np.dot(np.dot(H, P_pre), H.T) + alpha * R_1s))
                zNew = z_mea[:, t] - np.dot(H, xFault_pre)
                x_fault[:, t] = xFault_pre + np.dot(K, zNew)
                P0 = np.dot((I - np.dot(K, H)), P_pre)
        
        # 画图
        filter_plot(x_state, z_mea, x_KF, None, None, None, x_fault, T, mode, save_fig, mode_str)

    # 04-4-2 习题4（2）
    elif mode == 'Problem 4-2':
        mode_str = 'FT2'
        R_1s = 100  # 量测噪声方差为100m^2
        R_100s = 10  # 量测噪声方差为10m^2
        
        # 构造量测噪声序列
        V = np.zeros((1, T))
        V[0, :100] = np.sqrt(R_1s) * np.random.randn(100)  # 1-100s，方差为100m^2
        V[0, 100:] = np.sqrt(R_100s) * np.random.randn(T-100)  # 100s之后，方差为10m^2
        
        # 根据题设条件，用模型模拟真实状态
        for t in range(1, T):
            z_mea[:, t] = np.dot(H, x_state[:, t]) + V[0, t]
        
        # 再做一次卡尔曼滤波
        P0_kf = np.diag([10**2, 1**2])  # 重新设置初始P0
        for t in range(1, T):
            # 时间更新
            xKF_pre = np.dot(Phi, x_KF[:, t-1])
            P_pre = np.dot(np.dot(Phi, P0_kf), Phi.T) + np.dot(np.dot(Gamma, Q), Gamma.T)
            
            # 量测更新
            K = np.dot(np.dot(P_pre, H.T), np.linalg.pinv(np.dot(np.dot(H, P_pre), H.T) + R_1s))
            x_KF[:, t] = xKF_pre + np.dot(K, (z_mea[:, t] - np.dot(H, xKF_pre)))
            P0_kf = np.dot((I - np.dot(K, H)), P_pre)
        
        # 分配空间并赋予滤波器变量初值
        x_fault = np.zeros((nS, T))  # 自适应遗忘滤波状态值
        x_fault[:, 0] = x_state[:, 0]  # 自适应遗忘滤波器赋估计初值
        beta = np.zeros(T)
        beta[0] = 1
        b = 0.999  # 渐消因子
        C = np.dot(np.dot(H, (np.dot(np.dot(Phi, P0), Phi.T) + np.dot(np.dot(Gamma, Q), Gamma.T))), H.T) + R  # 新息序列方差阵
        zNew = 15
        
        # 量测噪声方差自适应滤波处理
        for t in range(1, T):
            # 时间更新
            xFault_pre = np.dot(Phi, x_fault[:, t-1])  # 状态一步预测
            P_pre = np.dot(np.dot(Phi, P0), Phi.T) + np.dot(np.dot(Gamma, Q), Gamma.T)  # 协方差一步预测
            
            # 量测更新
            beta[t] = beta[t-1] / (beta[t-1] + b)  # 更新新息序列方差阵的系数
            C = (1 - beta[t]) * C + beta[t] * (zNew * zNew)
            
            if abs(np.trace(C) / np.trace(np.dot(np.dot(H, P_pre), H.T) + R_1s)) > 2:  # 量测故障检测与隔离
                x_fault[:, t] = xFault_pre  # 无需量测更新，估计值用时间更新值代替
                P0 = P_pre
            else:
                alpha = np.trace(C - np.dot(np.dot(H, P_pre), H.T)) / np.trace(np.array([[R_1s]])) # 将 R_1s 转换为 1x1 矩阵
                K = np.dot(np.dot(P_pre, H.T), np.linalg.pinv(np.dot(np.dot(H, P_pre), H.T) + alpha * R_1s))
                zNew = z_mea[:, t] - np.dot(H, xFault_pre)
                x_fault[:, t] = xFault_pre + np.dot(K, zNew)
                P0 = np.dot((I - np.dot(K, H)), P_pre)
        
        # 画图
        filter_plot(x_state, z_mea, x_KF, None, None, None, x_fault, T, mode, save_fig, mode_str)

    # 保存数据
    if save_data:
        data_to_save = {
            'x_state': x_state,
            'z_mea': z_mea,
            'xKF': x_KF,
            'mode': mode,
            'T': T,
            'Ts': Ts,
            'Q': Q,
            'R': R
        }
        
        if mode == 'Problem 1':
            data_to_save['xIF'] = x_IF
        elif mode == 'Problem 2':
            data_to_save['xUD'] = x_UD
        elif mode == 'Problem 3':
            data_to_save['xFD'] = x_FD
        elif mode in ['Problem 4-1', 'Problem 4-2']:
            data_to_save['xFault'] = x_fault
        
        if with_time_flag:
            # 获取当前日期
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")
            save_path = f'./data/{mode_str}_{date_str}.pkl'
        else:
            save_path = f'./data/{mode_str}.pkl'
        
        # 使用pickle保存数据
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"数据已保存至: {save_path}")

if __name__ == '__main__':
    main()