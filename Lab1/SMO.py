import warnings
import numpy as np
from typing import Callable, Union
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class SMO:
    def __init__(self, C: float, tol: float = 1e-3,
                 max_passes: int = 10, kernel: Union[str, Callable] = 'linear',
                 kernel_params: dict = None, class_weight=None):
        """
        改进版SMO算法实现

        参数:
            X: 输入特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,), 值应为±1
            C: 惩罚系数 (必须为正数)
            tol: 容忍度 (用于确定是否违反KKT条件)
            max_passes: 最大迭代次数 (连续多少次迭代未改变alpha对 即停止)
            kernel: 核函数 ('linear', 'rbf', 'poly' 或自定义可调用函数)
            kernel_params: 核函数参数 (如gamma对于RBF核)
        """
        # 输入验证
        self.X = None
        self.y = None
        self._validate_input(C, tol, max_passes, class_weight)
        self.K = None
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel_params = kernel_params or {}
        self.kernel_name = kernel

        # 初始化模型参数
        self.b = 0.0
        self.class_weight = class_weight

    def _validate_Xy_input(self):
        if not isinstance(self.X, np.ndarray):
            warnings.warn("X should be a numpy array. Converting to numpy array.")
            self.X = np.array(self.X)
        if not isinstance(self.y, np.ndarray):
            warnings.warn("y should be a numpy array. Converting to numpy array.")
            self.y = np.array(self.y)

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.y.ndim != 1:
            self.y = self.y.reshape(-1)
            warnings.warn("y must be a 1D array, reshaping it")
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same length")

        unique_labels = set(np.unique(self.y))
        if unique_labels != {1, -1}:
            raise ValueError("y labels must be +1 or -1")

    def _validate_input(self, C, tol, max_passes, class_weight):
        """验证输入参数的有效性"""
        if class_weight is not None:
            if not isinstance(class_weight, dict):
                raise ValueError("class_weight must be a dictionary {class_label: weight}")
            if set(class_weight.keys()) != {1, -1}:
                raise ValueError("class_weight keys must be +1 and -1")

        if not isinstance(C, (int, float)) or C <= 0:
            raise ValueError("C must be a positive number")
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("tol must be a positive number")
        if not isinstance(max_passes, int) or max_passes <= 0:
            raise ValueError("max_passes must be a positive integer")

    def _init_kernel_matrix(self):
        n_samples = self.X.shape[0]
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                self.K[i, j] = self.kernel(self.X[i], self.X[j])
                self.K[j, i] = self.K[i, j]  # 对称性

    def _setup_kernel(self, kernel):
        """设置核函数"""
        if callable(kernel):
            self.kernel = kernel
        elif kernel == 'linear':
            self.kernel = self._linear_kernel
        elif kernel == 'rbf':  # gamma参数，get "auto","scale",fixed value
            self.kernel = self._rbf_kernel
            gamma = self.kernel_params.get('gamma', 'auto')
            if gamma == 'scale':
                self.gamma = 1.0 / (self.X.shape[1] * np.var(self.X))
            elif gamma == 'auto':
                self.gamma = 1.0 / self.X.shape[1]
            else:
                self.gamma = float(gamma)

        elif kernel == 'poly':
            self.kernel = self._polynomial_kernel
            self.degree = self.kernel_params.get('degree', 3)
            self.coef0 = self.kernel_params.get('coef0', 1.0)
        else:
            raise ValueError("kernel must be 'linear', 'rbf', 'poly' or a callable function")

    def _linear_kernel(self, x1, x2):
        """线性核函数"""
        return np.dot(x1, x2)

    def _rbf_kernel(self, x1, x2):
        """RBF核函数"""
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def _polynomial_kernel(self, x1, x2):
        """多项式核函数"""
        return (np.dot(x1, x2) + self.coef0) ** self.degree

    def fit(self, X, y):
        """训练SVM模型"""
        self.X = X
        self.y = y
        self._validate_Xy_input()

        self.alphas = np.zeros(len(X), dtype=np.float64)  # 拉格朗日乘子，\alpha_i \in [0,C]，C为错误分类容忍度
        # 设置核函数
        self._setup_kernel(self.kernel_name)
        self._init_kernel_matrix()

        passes = 0
        while passes < self.max_passes:  # 未达到最大未更新的迭代次数
            num_changed = 0
            # 遍历所有样本
            for i in range(len(self.alphas)):
                # 检查样本i是否违反KKT条件
                Ei = self._compute_error(i)
                if self._violates_kkt(i, Ei):
                    # 选择第二个alpha
                    j, Ej = self._select_second_alpha(i, Ei)

                    # 保存旧值
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

                    # 计算边界L和H
                    L, H = self._compute_bounds(i, j)
                    if L == H:
                        continue

                    # 计算eta，计算二阶导近似
                    eta = self._compute_eta(i, j)
                    if eta >= 0:
                        continue

                    # 更新alpha_j
                    self.alphas[j] = alpha_j_old - self.y[j] * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)  # \alphas_j \in [L, H] clip

                    # 检查alpha_j是否有足够大的变化
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # 更新alpha_i
                    self.alphas[i] = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

                    # 更新b
                    self._update_bias(i, j, Ei, Ej, alpha_i_old, alpha_j_old)

                    num_changed += 1

            # 检查是否收敛
            if num_changed == 0:
                passes += 1
            else:
                passes = 0

    def _violates_kkt(self, i, Ei):
        """检查样本i是否违反KKT条件
        \alpha_i = 0 ==> y_i f(x_i) > 1  or increase
        0 < \alpha_i < C  ==> y_i f(x_i) = 1 or adjust
        \alpha_i = C ==> y_i f(x_i) < 1 or decrease
        """
        y_i = self.y[i]
        C_i = self._get_weighted_C(y_i)  # 获取加权后的C值
        # 由于Ei = f(xi)-y_i, and y_i = {-1,1} thus y_i^2 = 1
        r = Ei * y_i  # 等价于f(xi)*y_i - 1
        return (r < -self.tol and self.alphas[i] < C_i) or (r > self.tol and self.alphas[i] > 0)

    def _get_weighted_C(self, y_i):
        """获取考虑类别权重后的C值"""
        if self.class_weight is None:
            return self.C
        return self.C * self.class_weight[y_i]

    def _select_second_alpha(self, i, Ei):
        """启发式选择第二个alpha"""
        if isinstance(Ei, (np.ndarray, list)):
            Ei = float(Ei[0]) if len(Ei) > 0 else 0.0

        # 优先选择非边界样本(0 < alpha < C)
        non_bound = [idx for idx in range(len(self.alphas)) if 0 < self.alphas[idx] < self.C]
        if non_bound:
            # 选择使|Ei-Ej|最大的样本
            j = max(non_bound, key=lambda j: abs(Ei - self._compute_error(j)))
            Ej = self._compute_error(j)
            return j, Ej

        # 如果没有非边界样本，随机选择
        j = np.random.choice([idx for idx in range(len(self.alphas)) if idx != i])
        Ej = self._compute_error(j)
        return j, Ej

    def _compute_bounds(self, i, j):
        """计算alpha_j的边界L和H（考虑类别权重）
        分为对称情况y_i==y_j和非对称情况
        对称情况：对约束 \alpha_i^new + \alpha_j^new = \alpha_i + \alpha_j = k
        """
        C_i = self._get_weighted_C(self.y[i])
        C_j = self._get_weighted_C(self.y[j])

        if self.y[i] == self.y[j]:
            L = max(0, self.alphas[i] + self.alphas[j] - C_j)
            H = min(C_i, self.alphas[i] + self.alphas[j])
        else:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(C_j, C_j + self.alphas[j] - self.alphas[i])
        return L, H

    def _compute_eta(self, i, j):
        """计算eta = 2K[i,j] - K[i,i] - K[j,j]"""
        if self.K is not None:
            return 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
        else:
            x_i, x_j = self.X[i], self.X[j]
            return 2 * self.kernel(x_i, x_j) - self.kernel(x_i, x_i) - self.kernel(x_j, x_j)

    def _update_bias(self, i, j, Ei, Ej, alpha_i_old, alpha_j_old):
        """更新偏置项b"""
        # 确保Ei, Ej是标量
        Ei = float(Ei[0]) if isinstance(Ei, (np.ndarray, list)) else float(Ei)
        Ej = float(Ej[0]) if isinstance(Ej, (np.ndarray, list)) else float(Ej)

        if self.K is not None:
            K_ii, K_jj, K_ij = self.K[i, i], self.K[j, j], self.K[i, j]
        else:
            x_i, x_j = self.X[i], self.X[j]
            K_ii, K_jj, K_ij = self.kernel(x_i, x_i), self.kernel(x_j, x_j), self.kernel(x_i, x_j)

        b1 = self.b - Ei - self.y[i] * (self.alphas[i] - alpha_i_old) * K_ii - self.y[j] * (
                self.alphas[j] - alpha_j_old) * K_ij
        b2 = self.b - Ej - self.y[i] * (self.alphas[i] - alpha_i_old) * K_ij - self.y[j] * (
                self.alphas[j] - alpha_j_old) * K_jj

        if 0 < self.alphas[i] < self.C:
            self.b = b1
        elif 0 < self.alphas[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        self.b = float(self.b)  # 确保b是标量

    def _compute_error(self, i):
        """
        误差计算：对样本点i计算对应误差E_i = f(x_i)-y_i
        w_i = \sum_{j=1} \alpha_j y_j x_j
        f(x_i) = w_i \cdot x_i + b
        """
        y_i = self.y[i]
        if self.K is not None:
            return float(np.dot(self.alphas * self.y, self.K[:, i]) + self.b - y_i)
        else:
            x_i = self.X[i]
            return float(sum(self.alphas * self.y *
                             np.array([self.kernel(x_j, x_i) for x_j in self.X])) + self.b - y_i)

    def predict(self, X_test):
        """
        预测样本类别
        参数:
            X_test: 测试样本 (n_samples, n_features)
            （计算测试样本与所有支持向量的加权核函数值之和）
        返回:
            pred: 预测标签 (n_samples,)
        """
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)

        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        if X_test.shape[1] != self.X.shape[1]:
            raise ValueError(f"Input shape must be (n_samples, {self.X.shape[1]})")

        pred = np.zeros(len(X_test))
        for i in range(len(X_test)):
            s = 0
            for j in range(len(self.X)):
                s += self.alphas[j] * self.y[j] * self.kernel(self.X[j], X_test[i])
            pred[i] = np.sign(s + self.b)

        return pred

    def get_support_vectors(self):
        """获取支持向量"""
        sv_indices = np.where(self.alphas > 1e-5)[0]
        return self.X[sv_indices], self.alphas[sv_indices], self.y[sv_indices]


import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def plot_decision_boundary(model, X, y, title, subplot):
    h = 0.02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    data = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(data)
    Z = Z.reshape(xx.shape)
    subplot.contourf(xx, yy, Z, alpha=0.5, cmap="coolwarm")
    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors='k')
    subplot.set_title(title)

def test_on_dataset(X, y, dataset_name, kernel):
    print(f"\n===== Dataset: {dataset_name} =====")
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"\n--- Kernel: {kernel} ---")

    # PCA 降到二维用于可视化
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    smo = SMO(C=1.0, tol=1e-5, max_passes=5, kernel=kernel, kernel_params={'degree': 3, 'gamma': 0.5})
    smo.fit(X_train_2d, y_train)
    y_pred_smo = smo.predict(X_test_2d)
    acc_smo = accuracy_score(y_test, y_pred_smo)
    print(f"Custom SMO Accuracy: {acc_smo:.4f}")

    svc = SVC(C=1.0, kernel=kernel, degree=3, gamma=0.5)
    svc.fit(X_train_2d, y_train)
    y_pred_svc = svc.predict(X_test_2d)
    acc_svc = accuracy_score(y_test, y_pred_svc)
    print(f"Sklearn SVC Accuracy: {acc_svc:.4f}")

    # 可视化决策边界对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundary(smo, X_train_2d, y_train, f"Custom SMO ({kernel})", axes[0])
    plot_decision_boundary(svc, X_train_2d, y_train, f"Sklearn SVC ({kernel})", axes[1])
    fig.suptitle(f"Decision Boundary on {dataset_name} - {kernel} kernel", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"smo_with_{kernel}.jpg",dpi=600)
    plt.show()

if __name__ == '__main__':
    # 1. Iris
    data = load_iris()
    X_bc, y_bc = data.data, data.target
    test_on_dataset(X_bc, y_bc, "Iris", kernel="linear")

    # 2. Moons 数据集（非线性）
    X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)
    test_on_dataset(X_moons, y_moons, "Make Moons", kernel="poly")

    # # 设置随机种子
    # np.random.seed(42)
    #
    # # 每个类别的样本数量
    # num_samples = 200
    # # 噪声程度
    # noise = 0.3
    #
    # # 生成类别1的数据（内圆）
    # radius_inner = 1.0
    # angles_inner = 2 * np.pi * np.random.rand(num_samples)
    # x1_inner = radius_inner * np.cos(angles_inner) + np.random.normal(0, noise, num_samples)
    # y1_inner = radius_inner * np.sin(angles_inner) + np.random.normal(0, noise, num_samples)
    # X1_inner = np.vstack((x1_inner, y1_inner)).T
    #
    # # 生成类别2的数据（外圆）
    # radius_outer = 2.0
    # angles_outer = 2 * np.pi * np.random.rand(num_samples)
    # x2_outer = radius_outer * np.cos(angles_outer) + np.random.normal(0, noise, num_samples)
    # y2_outer = radius_outer * np.sin(angles_outer) + np.random.normal(0, noise, num_samples)
    # X2_outer = np.vstack((x2_outer, y2_outer)).T
    #
    # # 将X1_inner和X2_outer在垂直方向拼接，形成最终的特征矩阵X，形状为(400, 2)
    # X = np.vstack((X1_inner, X2_outer))
    #
    # # 创建标签向量y，前200个为1（类别1），后200个为-1（类别2），形状为(400,)
    # y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
    #
    # # 将数据分为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #
    # # 创建SVC模型并进行训练
    # model_svc = SVC(kernel='rbf', C=1, gamma='auto')
    # model_svc.fit(X_train, y_train)
    #
    # # 对测试集进行预测
    # y_pred = model_svc.predict(X_test)
    #
    # # 输出模型的准确率
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'SVC模型准确率: {accuracy:.4f}')
    #
    # model_smo = SMO(kernel='rbf', C=1)
    # model_smo.fit(X_train, y_train)
    #
    # # 对测试集进行预测
    # y_pred = model_smo.predict(X_test)
    #
    # # 输出模型的准确率
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'SMO模型准确率: {accuracy:.4f}')
    #
    # # 创建 1x2 子图，可视化决策边界
    # fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    #
    # xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
    #                      np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
    #
    # # 绘制 SVC 决策边界
    # Z_svc = model_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z_svc = Z_svc.reshape(xx.shape)
    # axes[0].contourf(xx, yy, Z_svc, alpha=0.8, cmap='coolwarm')
    # axes[0].scatter(X1_inner[:, 0], X1_inner[:, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
    # axes[0].scatter(X2_outer[:, 0], X2_outer[:, 1], c='red', label='类别2', alpha=0.6, edgecolors='k')
    # axes[0].set_title('SVC 决策边界')
    # axes[0].set_xlabel('x')
    # axes[0].set_ylabel('y')
    # axes[0].legend()
    # axes[0].grid(True)
    # axes[0].axis('equal')
    #
    # # 绘制 SMO 决策边界
    # Z_smo = model_smo.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z_smo = Z_smo.reshape(xx.shape)
    # axes[1].contourf(xx, yy, Z_smo, alpha=0.8, cmap='coolwarm')
    # axes[1].scatter(X1_inner[:, 0], X1_inner[:, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
    # axes[1].scatter(X2_outer[:, 0], X2_outer[:, 1], c='red', label='类别2', alpha=0.6, edgecolors='k')
    # axes[1].set_title('SMO 决策边界')
    # axes[1].set_xlabel('x')
    # axes[1].set_ylabel('y')
    # axes[1].legend()
    # axes[1].grid(True)
    # axes[1].axis('equal')
    #
    # # 显示图像
    # plt.tight_layout()
    # plt.savefig("SMOVSSVC.jpg", dpi=600)
    # plt.show()
