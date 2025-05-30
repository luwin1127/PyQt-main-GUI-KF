import numpy as np
import matplotlib.pyplot as plt


def filter_plot(x_state, z_mea, x_KF, x_IF, x_UD, x_FD, x_FT, T, mode, save_fig, save_mode):
    # 设置中文字体支持
    plt.rcParams["font.family"] = ["STSONG"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
    t_plot = np.arange(1, T + 1)

    # 图1
    fig1 = plt.figure(figsize=(5.5, 4.5), facecolor='white')
    plt.title('真实位置与滤波位置的比较')
    plt.xlabel('时间 t/s')
    plt.ylabel('位置误差 m')
    plt.grid(True)
    plt.plot(t_plot, x_state[0, :] - z_mea[0, :], 'b-', linewidth=1.5)
    plt.plot(t_plot, x_state[0, :] - x_KF[0, :], 'r-', linewidth=1.5)
    if mode == 'Problem 1':
        plt.plot(t_plot, x_state[0, :] - x_IF[0, :], 'g-', linewidth=1.5)
        plt.legend(['测量位置误差', '滤波位置误差(KF)', '滤波位置误差(IF)'])
        plt.axis([0, T, -150, 150])
    elif mode == 'Problem 2':
        plt.plot(t_plot, x_state[0, :] - x_UD[0, :], 'c-', linewidth=1.5)
        plt.legend(['测量位置误差', '滤波位置误差(KF)', '滤波位置误差(UD)'])
    elif mode == 'Problem 3':
        plt.plot(t_plot, x_state[0, :] - x_FD[0, :], 'm-', linewidth=1.5)
        plt.legend(['测量位置误差', '滤波位置误差(KF)', '滤波位置误差(FD)'])
    elif mode in ['Problem 4-1', 'Problem 4-2']:
        plt.plot(t_plot, x_state[0, :] - x_FT[0, :],
                 color=[0, 0.61, 0.46], linewidth=1.5)
        plt.legend(['测量位置误差', '滤波位置误差(KF)', '滤波位置误差(FT)'])
    if save_fig:
        save_str = f'./figures/01-真实位置与滤波位置的比较-{save_mode}.png'
        plt.savefig(save_str)

    # 图2
    fig2 = plt.figure(figsize=(5.5, 4.5), facecolor='white')
    plt.title('真实速度与滤波速度的比较')
    plt.xlabel('时间 t/s')
    plt.ylabel('速度误差 m/s')
    plt.grid(True)
    plt.plot(t_plot, x_state[1, :] - x_KF[1, :], 'r-', linewidth=1.5)
    if mode == 'Problem 1':
        plt.plot(t_plot, x_state[1, :] - x_IF[1, :], 'g-', linewidth=1.5)
        plt.legend(['速度误差(KF)', '速度误差(IF)'])
        plt.axis([0, T, -10, 10])
    elif mode == 'Problem 2':
        plt.plot(t_plot, x_state[1, :] - x_UD[1, :], 'c-', linewidth=1.5)
        plt.legend(['速度误差(KF)', '速度误差(UD)'])
    elif mode == 'Problem 3':
        plt.plot(t_plot, x_state[1, :] - x_FD[1, :], 'm-', linewidth=1.5)
        plt.legend(['速度误差(KF)', '速度误差(FD)'])
    elif mode in ['Problem 4-1', 'Problem 4-2']:
        plt.plot(t_plot, x_state[1, :] - x_FT[1, :],
                 color=[0, 0.61, 0.46], linewidth=1.5)
        plt.legend(['速度误差(KF)', '速度误差(FT)'])
    if save_fig:
        save_str = f'./figures/02-真实速度与滤波速度的比较-{save_mode}.png'
        plt.savefig(save_str)

    # 图3
    fig3 = plt.figure(figsize=(5.5, 4.5), facecolor='white')
    plt.title('真实位置、测量位置与滤波位置的比较')
    plt.xlabel('时间 t/s')
    plt.ylabel('位置 m')
    plt.grid(True)
    plt.plot(t_plot, x_state[0, :], 'k-', linewidth=1.5)
    plt.plot(t_plot, z_mea[0, :], 'b-', linewidth=1.5)
    plt.plot(t_plot, x_KF[0, :], 'r-', linewidth=1.5)
    if mode == 'Problem 1':
        plt.plot(t_plot, x_IF[0, :], 'g-', linewidth=1.5)
        plt.legend(['真实位置', '测量位置', '滤波位置(KF)', '滤波位置(IF)'])
    elif mode == 'Problem 2':
        plt.plot(t_plot, x_UD[0, :], 'c-', linewidth=1.5)
        plt.legend(['真实位置', '测量位置', '滤波位置(KF)', '滤波位置(UD)'])
    elif mode == 'Problem 3':
        plt.plot(t_plot, x_FD[0, :], 'm-', linewidth=1.5)
        plt.legend(['真实位置', '测量位置', '滤波位置(KF)', '滤波位置(FD)'])
    elif mode in ['Problem 4-1', 'Problem 4-2']:
        plt.plot(t_plot, x_FT[0, :], color=[0, 0.61, 0.46], linewidth=1.5)
        plt.legend(['真实位置', '测量位置', '滤波位置(KF)', '滤波位置(FT)'])
    ax = fig3.add_axes([0.25, 0.25, 0.25, 0.25])
    ax.plot(t_plot, x_state[0, :], 'k-', linewidth=1.5)
    ax.plot(t_plot, z_mea[0, :], 'b-', linewidth=1.5)
    ax.plot(t_plot, x_KF[0, :], 'r-', linewidth=1.5)
    if mode == 'Problem 1':
        ax.plot(t_plot, x_IF[0, :], 'g-', linewidth=1.5)
    elif mode == 'Problem 2':
        ax.plot(t_plot, x_UD[0, :], 'c-', linewidth=1.5)
    elif mode == 'Problem 3':
        ax.plot(t_plot, x_FD[0, :], 'm-', linewidth=1.5)
    elif mode in ['Problem 4-1', 'Problem 4-2']:
        ax.plot(t_plot, x_FT[0, :], color=[0, 0.61, 0.46], linewidth=1.5)
    ax.set_xlim([T / 2, T / 2 + 0.5])
    if save_fig:
        save_str = f'./figures/03-真实位置、测量位置与滤波位置的比较-{save_mode}.png'
        plt.savefig(save_str)

    # 图4
    fig4 = plt.figure(figsize=(5.5, 4.5), facecolor='white')
    plt.title('真实速度与滤波速度的比较')
    plt.xlabel('时间 t/s')
    plt.ylabel('速度 m/s')
    plt.grid(True)
    plt.plot(t_plot, x_state[1, :], 'b-', linewidth=1.5)
    plt.plot(t_plot, x_KF[1, :], 'r-', linewidth=1.5)
    if mode == 'Problem 1':
        plt.plot(t_plot, x_IF[1, :], 'g-', linewidth=1.5)
        plt.legend(['真实速度', '滤波速度(KF)', '滤波速度(IF)'])
        plt.axis([0, T, -310, -290])
    elif mode == 'Problem 2':
        plt.plot(t_plot, x_UD[1, :], 'c-', linewidth=1.5)
        plt.legend(['真实速度', '滤波速度(KF)', '滤波速度(UD)'])
    elif mode == 'Problem 3':
        plt.plot(t_plot, x_FD[1, :], 'm-', linewidth=1.5)
        plt.legend(['真实速度', '滤波速度(KF)', '滤波速度(FD)'])
    elif mode in ['Problem 4-1', 'Problem 4-2']:
        plt.plot(t_plot, x_FT[1, :], color=[0, 0.61, 0.46], linewidth=1.5)
        plt.legend(['真实速度', '滤波速度(KF)', '滤波速度(FT)'])
    if save_fig:
        save_str = f'./figures/04-真实速度与滤波速度的比较-{save_mode}.png'
        plt.savefig(save_str)

    # 图5
    fig5 = plt.figure(figsize=(5.5, 4.5), facecolor='white')
    plt.subplot(211)
    plt.title('滤波位置与速度')
    plt.xlabel('时间 t/s')
    plt.ylabel('滤波位置 m')
    plt.plot(t_plot, x_KF[0, :], 'r-', linewidth=1.5)
    if mode == 'Problem 1':
        plt.plot(t_plot, x_IF[0, :], 'g-', linewidth=1.5)
        plt.legend(['KF', 'IF'])
    elif mode == 'Problem 2':
        plt.plot(t_plot, x_UD[0, :], 'c-', linewidth=1.5)
        plt.legend(['KF', 'UD'])
    elif mode == 'Problem 3':
        plt.plot(t_plot, x_FD[0, :], 'm-', linewidth=1.5)
        plt.legend(['KF', 'FD'])
    elif mode in ['Problem 4-1', 'Problem 4-2']:
        plt.plot(t_plot, x_FT[0, :], color=[0, 0.61, 0.46], linewidth=1.5)
        plt.legend(['KF', 'FT'])
    plt.subplot(212)
    plt.xlabel('时间 t/s')
    plt.ylabel('滤波速度 m/s')
    plt.plot(t_plot, x_KF[1, :], 'r-', linewidth=1.5)
    if mode == 'Problem 1':
        plt.plot(t_plot, x_IF[1, :], 'g-', linewidth=1.5)
        plt.legend(['KF', 'IF'])
        plt.axis([0, T, -310, -290])
    elif mode == 'Problem 2':
        plt.plot(t_plot, x_UD[1, :], 'c-', linewidth=1.5)
        plt.legend(['KF', 'UD'])
    elif mode == 'Problem 3':
        plt.plot(t_plot, x_FD[1, :], 'm-', linewidth=1.5)
        plt.legend(['KF', 'FD'])
    elif mode in ['Problem 4-1', 'Problem 4-2']:
        plt.plot(t_plot, x_FT[1, :], color=[0, 0.61, 0.46], linewidth=1.5)
        plt.legend(['KF', 'FT'])
    if save_fig:
        save_str = f'./figures/05-滤波位置与速度-{save_mode}.png'
        plt.savefig(save_str)

    if not save_fig:
        plt.show()
    else:
        # 确保关闭所有图形窗口
        plt.close('all')