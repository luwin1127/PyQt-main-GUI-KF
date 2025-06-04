import sys
import os
import numpy as np
# 导入PyQt5所需模块
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from Ui_mainGUI_alpha import Ui_MainWindow
# 导入PyQt风格模块

# 导入方法
from user_function.KF import KF
from user_function.UD_update import UD_update
from user_function.udu import udu
from user_function.filter_plot import filter_plot
# 导入保存数据所需模块
import pickle
import numpy as np
from datetime import datetime

class MainApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        """主窗口初始化"""
        super().__init__()
        self.setupUi(self)          # 加载UI设计
        self.init_ui()              # 初始化界面设置
        self.bind_events()          # 绑定UI事件处理函数
        self.init_flags()           # 初始化标志位变量
        
        # 设置图形显示区域无边框和滚动条
        self.eq1_disp.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.eq1_disp.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.eq2_disp.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.eq2_disp.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        # 设置默认选中状态
        self.yes_btn.setChecked(True)       # 默认选择"带时间戳"选项
        self.save_fig_box.setChecked(False)  # 默认选择"保存图片"选项
        self.save_data_box.setChecked(True) # 默认选择"保存数据"选项

        # 初始化下拉菜单栏状态
        self.method_menu.currentIndexChanged.emit(0)  # 触发索引0的变化（默认选项）
        
        # 设置应用程序图标
        self.set_application_icon()

        # 绑定保存图片复选框的状态变更事件
        self.save_fig_box.stateChanged.connect(self.check_save_fig_warning)

    def init_ui(self):
        """初始化界面显示设置"""
        self.setWindowTitle("卡尔曼滤波仿真软件Demo")  # 设置窗口标题
        self.update_graph_displays(0)       # 显示默认滤波方法对应的公式图

    def init_flags(self):
        """初始化控制标志位，存储UI选择状态"""
        self.method_flag = ""               # 当前选择的滤波方法
        self.save_fig_flag = False          # 是否保存图表
        self.save_data_flag = False         # 是否保存数据
        self.with_time_flag = False         # 是否在保存数据时添加时间戳

    def bind_events(self):
        """绑定UI元素事件到处理函数"""
        self.exit_btn.clicked.connect(self.close)                               # 退出按钮
        self.simu_btn.clicked.connect(self.run_simulation)                      # 仿真按钮
        self.method_menu.currentIndexChanged.connect(self.update_method_flag)   # 滤波方法选择
        self.save_fig_box.toggled.connect(self.update_save_fig_flag)            # 保存图表复选框
        self.save_data_box.toggled.connect(self.update_save_data_flag)          # 保存数据复选框
        self.yes_btn.toggled.connect(self.update_with_time_flag)                # 带时间戳单选按钮
        self.no_btn.toggled.connect(self.update_with_time_flag)                 # 不带时间戳单选按钮

    def set_application_icon(self):
        """设置应用程序图标"""
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asserts', 'logo.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
            print(f"应用程序图标已设置: {icon_path}")
        else:
            print(f"警告: 图标文件不存在 - {icon_path}")

    def check_save_fig_warning(self, state):
        """检测保存图片复选框状态，显示提示"""
        if state == QtCore.Qt.Checked:  # 当勾选时
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("选择“保存图片”则仿真完毕之后不会展示结果。")
            msg.setWindowTitle("操作提示")
            msg.exec_()

    def update_method_flag(self, index):
        """更新滤波方法标志位并刷新公式图显示"""
        # 滤波方法索引到名称的映射
        method_map = {
            0: "请选择滤波方式",
            1: "信息滤波",
            2: "UD滤波",
            3: "遗忘滤波",
            4: "自适应遗忘滤波(1)",
            5: "自适应遗忘滤波(2)"
        }
        self.method_flag = method_map.get(index, "请选择滤波方式")  # 更新方法标志位
        self.update_graph_displays(index)             # 刷新公式图显示

    def update_save_fig_flag(self, state):
        """更新保存图表标志位"""
        self.save_fig_flag = state

    def update_save_data_flag(self, state):
        """更新保存数据标志位"""
        self.save_data_flag = state

    def update_with_time_flag(self, state):
        """更新保存数据时是否添加时间戳标志位"""
        self.with_time_flag = self.yes_btn.isChecked()  # 根据Yes按钮状态更新标志

    def update_graph_displays(self, index):
        """根据选择的滤波方法更新公式图显示"""
        # 滤波方法索引到公式图路径的映射
        image_paths = [
            ("./asserts/KF1.png", "./asserts/KF2.png"),    # 0: 卡尔曼滤波
            ("./asserts/IF1.png", "./asserts/IF2.png"),    # 1: 信息滤波
            ("./asserts/UD1.png", "./asserts/UD2.png"),    # 2: UD滤波
            ("./asserts/FD1.png", "./asserts/FD2.png"),    # 3: 遗忘滤波
            ("./asserts/FT1.png", "./asserts/FT2.png"),    # 4: 自适应遗忘滤波(1)
            ("./asserts/FT1.png", "./asserts/FT2.png"),    # 5: 自适应遗忘滤波(2)
        ]
        
        if 0 <= index < len(image_paths):  # 确保索引有效
            img1_path, img2_path = image_paths[index]
            self.display_image(self.eq1_disp, img1_path)  # 显示第一个公式图
            self.display_image(self.eq2_disp, img2_path)  # 显示第二个公式图

    def display_image(self, graphics_view, image_path):
        """在指定的QGraphicsView中显示图片"""
        scene = QtWidgets.QGraphicsScene(graphics_view)  # 创建图形场景
        pixmap = QtGui.QPixmap(image_path)               # 加载图片
        
        # 缩放图片以适应视图大小，保持纵横比并平滑缩放
        scaled_pixmap = pixmap.scaled(
            graphics_view.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        
        scene.addPixmap(scaled_pixmap)  # 将缩放后的图片添加到场景
        graphics_view.setScene(scene)   # 设置视图的场景
        graphics_view.setAlignment(QtCore.Qt.AlignCenter)  # 图片居中显示

    def get_lineedit_values(self):
        """获取所有QLineEdit控件中的参数值"""
        return {
            "simu_time": self.simu_edit.text(),       # 仿真时长
            "sample_time": self.sample_edit.text(),   # 采样时长
            "init_state": self.init_edit.text(),      # 初始状态
            "process_noise": self.process_edit.text(),  # 过程噪声
            "P_matrix": self.P_edit.text(),           # 协方差矩阵
            "measure_noise": self.measure_edit.text(), # 量测噪声
            "info_matrix": self.info_edit.text()      # 信息矩阵
        }

    def parse_numeric_params(self, params):
        """解析并验证数值参数，转换为指定数据类型"""
        errors = []
        parsed = {}

        # 解析整数
        for key in ["simu_time"]:
            try:
                parsed[key] = int(params[key])
            except ValueError:
                errors.append(f"{key}: 请输入有效整数")
        
        # 解析浮点数参数
        for key in ["sample_time", "process_noise", "measure_noise"]:
            try:
                parsed[key] = float(params[key])
            except ValueError:
                errors.append(f"{key}: 请输入有效的浮点数")
        
        # 解析初始状态 (2,) 数组
        try:
            # 处理类似 "[10000;-300]" 或 "[10000, -300]" 的输入格式
            values = params["init_state"].strip("[]").replace(";", ",").split(",")
            parsed["init_state"] = np.array([float(v.strip()) for v in values], dtype=float) # v.strip()用于移除字符串开头和结尾的特定字符
            if parsed["init_state"].shape != (2,):
                raise ValueError("需要包含两个元素的数组")
        except Exception as e:
            errors.append(f"init_state: {str(e)} (格式应为 [值1, 值2])")
        
        # 解析协方差矩阵 (2,) 数组
        try:
            # 处理类似 "[100,1]" 或 "[100;1]" 的输入格式
            values = params["P_matrix"].strip("[]").replace(";", ",").split(",")
            parsed["P_matrix"] = np.array([float(v.strip()) for v in values], dtype=float) # v.strip()用于移除字符串开头和结尾的特定字符
            if parsed["P_matrix"].shape != (2,):
                raise ValueError("需要包含两个元素的数组")
        except Exception as e:
            errors.append(f"P_matrix: {str(e)} (格式应为 [值1, 值2])")
        
        # 解析信息矩阵 (2,2) 数组
        try:
            # 处理类似 "[0,0;0,0]" 的输入格式
            rows = params["info_matrix"].strip("[]").split(";")
            parsed["info_matrix"] = np.array(
                [[float(v) for v in row.split(",")] for row in rows], 
                dtype=float
            )
            if parsed["info_matrix"].shape != (2, 2):
                raise ValueError("需要2x2的数组格式")
        except Exception as e:
            errors.append(f"info_matrix: {str(e)} (格式应为 [值1,值2;值3,值4])")
        
        return parsed, errors

    def run_simulation(self):
        """执行仿真按钮点击事件处理"""
        # 收集标志位参数
        flags = {
            "method_flag": self.method_flag,
            "save_fig_flag": self.save_fig_flag,
            "save_data_flag": self.save_data_flag,
            "save_data_time_flag": self.with_time_flag
        }
        
        # 收集文本框参数
        params_txt = self.get_lineedit_values()
        
        # 解析数值参数
        params, errors = self.parse_numeric_params(params_txt)
        
        # 检查是否有解析错误
        if errors:
            error_msg = "参数解析错误:\n" + "\n".join(errors)
            QMessageBox.critical(self, "参数错误", error_msg)
            return
        
        # 打印所有参数（实际应用中可替换为仿真计算逻辑）
        # print("\n===== 仿真参数设置 =====")
        # print("滤波方法:", flags["method_flag"])
        # print("保存图表:", flags["save_fig_flag"])
        # print("保存数据:", flags["save_data_flag"])
        # print("数据带时间戳:", flags["save_data_time_flag"])
        
        # print("\n===== 仿真具体参数 =====")
        # print("仿真时长:", params["simu_time"])
        # print("采样时长:", params["sample_time"])
        # print("初始状态:\n", params["init_state"])
        # print("过程噪声:", params["process_noise"])
        # print("协方差矩阵:\n", params["P_matrix"])
        # print("量测噪声:", params["measure_noise"])
        # print("信息矩阵:\n", params["info_matrix"])
        
        print("===== 开始仿真计算 =====")

        # 仿真参数
        T = params["simu_time"]                 # 仿真时长
        Ts = params["sample_time"]              # 采样时长
        Q = params["process_noise"] * Ts        # 过程噪声
        R = params["measure_noise"]             # 量测噪声

        # 生成噪声序列
        W = np.sqrt(Q) * np.random.randn(1, T)
        V = np.sqrt(R) * np.random.randn(1, T)
        p0 = params["P_matrix"]                 # 初始协方差阵参数
        P0 = np.diag([p0[0], p0[1]])            # 初始协方差阵
        I0 = params["info_matrix"]              # 初始信息阵设置为0

        # 系统矩阵
        A = np.array([[0, 1], [0, 0]])          # 状态矩阵
        I = np.eye(2)                           # 单位阵
        Phi = I + A * Ts                        # 离散化
        H = np.array([[1, 0]])                  # 量测矩阵
        Gamma = np.array([[0], [1]])

        # 设置滤波维度
        nS = 2                                  # 状态维度
        nZ = 1                                  # 观测维度

        # 分配空间
        x_state = np.zeros((nS, T))             # 系统真实值
        z_mea = np.zeros((nZ, T))               # 系统观测值
        x_KF = np.zeros((nS, T))                # 卡尔曼滤波状态值

        # 赋初值
        x_state[:, 0] = params["init_state"]    # 系统状态初值
        z_mea[:, 0] = np.dot(H, x_state[:, 0])  # 系统观测初值
        x_KF[:, 0] = x_state[:, 0]              # 卡尔曼滤波器估计初值

        # 02 用模型模拟真实状态
        for t in range(1, T):
            x_state[:, t] = np.dot(Phi,x_state[:, t-1]) + np.dot(Gamma, W[0, t]).squeeze()
            z_mea[:, t] = np.dot(H, x_state[:, t]) + V[0, t]

        # 03-1 Kalman滤波
        if self.method_flag == '请选择滤波方式':
            mode = 'null'
            print("请先选择滤波方法！")
            QMessageBox.warning(self, "操作提示", "请先选择滤波方法！", QMessageBox.Ok)
            return  # 终止仿真流程
        
            # #下述代码为KF滤波方法
            # mode = 'null'
            # mode_str = 'KF'
            # P0_kf = P0.copy()  # 复制初始P0
            # for t in range(1, T):
            #     x_KF[:, t], P0_kf = KF(x_KF, P0_kf, z_mea, Phi, Gamma, H, Q, R, I, t)
            # # 画图
            # filter_plot(x_state, z_mea, x_KF, None, None, None, None, T, mode, self.save_fig_flag, mode_str)

        # 04-1 习题1：信息滤波
        elif self.method_flag == '信息滤波':
            mode = 'Problem 1'
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
            filter_plot(x_state, z_mea, x_KF, x_IF, None, None, None, T, mode, self.save_fig_flag, mode_str)
            print("滤波方法:", flags["method_flag"])

        # 04-2 习题2：UD滤波
        elif self.method_flag == 'UD滤波':
            mode = 'Problem 2'
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
            filter_plot(x_state, z_mea, x_KF, None, x_UD, None, None, T, mode, self.save_fig_flag, mode_str)
            print("滤波方法:", flags["method_flag"])

        
        # 04-3 习题3：遗忘滤波
        elif self.method_flag == '遗忘滤波':
            mode = 'Problem 3'
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
            filter_plot(x_state, z_mea, x_KF, None, None, x_FD, None, T, mode, self.save_fig_flag, mode_str)
            print("滤波方法:", flags["method_flag"])

        # 04-4-1 习题4（1）
        elif self.method_flag == '自适应遗忘滤波(1)':
            mode = 'Problem 4-1'
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
            filter_plot(x_state, z_mea, x_KF, None, None, None, x_fault, T, mode, self.save_fig_flag, mode_str)
            print("滤波方法:", flags["method_flag"])

        # 04-4-2 习题4（2）
        elif self.method_flag == '自适应遗忘滤波(2)':
            mode = 'Problem 4-2'
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
            filter_plot(x_state, z_mea, x_KF, None, None, None, x_fault, T, mode, self.save_fig_flag, mode_str)
            print("滤波方法:", flags["method_flag"])

        # 保存数据
        if self.save_data_flag:
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
            
            if self.with_time_flag:
                # 获取当前日期
                now = datetime.now()
                date_str = now.strftime("%Y%m%d")
                save_path = f'./data/{mode_str}_{date_str}.pkl'
            else:
                save_path = f'./data/{mode_str}.pkl'
            
            # 使用pickle保存数据
            with open(save_path, 'wb') as f:
                pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"仿真结果已保存至: ./figures/")
            print(f"数据已保存至: {save_path}")
        print("===== 仿真计算结束 =====")

if __name__ == '__main__':
    # NOTICE：01必须放在02前面
    # 01-高分辨率屏幕控件自适应调整
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) # 原有控件缩放，参考：https://zhuanlan.zhihu.com/p/401503085

    # 02-应用程序初始化
    app = QApplication(sys.argv)
    
    # 03-高分辨率屏幕字体自适应调整
    screen = app.primaryScreen()    # 返回当前主显示器的信息
    scale_factor = screen.logicalDotsPerInch() / 96  # 96dpi为标准缩放（100%），结果如1.75（175%缩放）
    font = QtGui.QFont()            # 创建默认字体
    font.setPointSize(int(10 * scale_factor))  # 原字体10pt，乘以缩放因子
    app.setFont(font)

    # 04-创建并显示应用窗口
    window = MainApp()
    window.show()
    sys.exit(app.exec_())