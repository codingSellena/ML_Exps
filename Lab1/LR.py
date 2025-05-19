import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class LogisticRegression:
    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6,C=0.1):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.l2_lambda = 1.0  # L2 正则化系数(仿照sklearn)
        self.theta = None
        self.loss_history = []


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, h, y):
        # L = -1/m * \sum[y log h + (1-y) log(1-h)]
        m = len(y)
        # 不包含偏置项的正则化项
        reg = (self.l2_lambda / (2 * m)) * np.sum(self.theta[1:] ** 2)
        loss = (-1 / m) * np.sum(y * np.log(h + 1e-9) + (1 - y) * np.log(1 - h + 1e-9))
        return loss + reg

    def fit(self, X, y, verbose=True):
        self._input_validate(X,y)
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X]) # X = [1, x1, x2, ... , xn]
        self.theta = np.zeros(n + 1)

        for i in range(self.max_iter):
            z = X @ self.theta
            h = self.sigmoid(z)
            loss = self.compute_loss(h, y)
            self.loss_history.append(loss)

            # 梯度计算，偏置项不加正则化
            # (1/m)* x^T @ (h-y)
            gradient = (X.T @ (h - y)) / m
            gradient[1:] += (self.l2_lambda / m) * self.theta[1:]

            prev_theta = self.theta.copy()
            self.theta -= self.lr * gradient

            if np.linalg.norm(self.theta - prev_theta) < self.tol:
                if verbose:
                    print(f"Model Converged at iteration {i}, Loss: {loss:.6f}")
                break

            if verbose and i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.6f}")

        else:
            warnings.warn("Maximum iterations reached without convergence.")

    def predict_proba(self, X):
        self._input_validate(X)
        m = X.shape[0]
        X = np.hstack([np.ones((m, 1)), X])
        return self.sigmoid(X @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


    def plot_loss(self):
        if len(self.loss_history) == 0:
            raise ValueError("Loss history is empty. Ensure that the model has been trained before plotting.")
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.show()

    def _input_validate(self, X, y=None):
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be a numpy ndarray or pandas DataFrame.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (samples x features).")

        if y is not None:
            if not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
                raise ValueError("y must be a numpy ndarray, pandas Series, or pandas DataFrame.")
            if isinstance(y, pd.DataFrame):
                y = y.values
            if y.ndim > 1:
                y = y.reshape(-1)
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Number of samples in X ({X.shape[0]}) must match number of labels in y ({y.shape[0]}).")



if __name__ == '__main__':
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.datasets import load_wine
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    # from sklearn.metrics import accuracy_score, classification_report
    # from sklearn.decomposition import PCA
    #
    #
    # # 1. 加载葡萄酒数据集
    # wine = load_wine()
    # X = wine.data
    # y = wine.target
    # feature_names = wine.feature_names
    # class_names = wine.target_names
    #
    # print(f"数据集形状: {X.shape}")
    # print(f"特征名: {feature_names}")
    # print(f"类别名: {class_names}")
    #
    # # 2. 数据预处理
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    #
    # # 使用PCA降维到2维以便可视化（但仍用所有特征训练模型）
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_scaled)
    #
    # # 3. 划分训练测试集
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    # X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    #
    # # 4. 自定义逻辑回归（OvR策略实现多分类）
    # custom_models = []
    # for cls in np.unique(y_train):
    #     y_binary = (y_train == cls).astype(int)
    #     model = LogisticRegression(lr=0.1, max_iter=1000)
    #     model.fit(X_train, y_binary)
    #     custom_models.append(model)
    #
    #
    # def predict_custom(X):
    #     scores = np.zeros((X.shape[0], len(custom_models)))
    #     for i, model in enumerate(custom_models):
    #         scores[:, i] = model.predict_proba(X)
    #     return np.argmax(scores, axis=1)
    #
    #
    # y_pred_custom = predict_custom(X_test)
    # acc_custom = accuracy_score(y_test, y_pred_custom)
    # print("\n✅ 自定义逻辑回归性能:")
    # print(f"准确率: {acc_custom:.4f}")
    # print(classification_report(y_test, y_pred_custom, target_names=class_names))
    #
    # # 5. sklearn逻辑回归
    # sklearn_lr = OneVsRestClassifier(SklearnLogisticRegression(solver='lbfgs', max_iter=1000))
    # sklearn_lr.fit(X_train, y_train)
    # y_pred_sklearn = sklearn_lr.predict(X_test)
    # acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    # print("\n✅ sklearn逻辑回归性能:")
    # print(f"准确率: {acc_sklearn:.4f}")
    # print(classification_report(y_test, y_pred_sklearn, target_names=class_names))
    #
    #
    # # 6. 决策边界可视化（基于PCA降维后的2D空间）
    # def plot_decision_boundary(predict_func, X_pca, y, title):
    #     h = 0.02  # 网格步长
    #     x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    #     y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    #
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                          np.arange(y_min, y_max, h))
    #
    #     # 将网格点逆变换回原始特征空间
    #     grid_pca = np.c_[xx.ravel(), yy.ravel()]
    #     grid_original = pca.inverse_transform(grid_pca)
    #
    #     # 预测
    #     Z = predict_func(grid_original)
    #     Z = Z.reshape(xx.shape)
    #
    #     plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    #     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap='viridis')
    #
    #     # 创建图例
    #     legend1 = plt.legend(*scatter.legend_elements(),
    #                         title="Classes", loc="upper right")
    #     plt.gca().add_artist(legend1)
    #
    #     plt.title(title)
    #     plt.xlabel('Feature 1 (标准化)')
    #     plt.ylabel('Feature 2 (标准化)')
    #
    #
    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 2, 1)
    # plot_decision_boundary(predict_custom, X_test_pca, y_test, '自定义逻辑回归决策边界\n(PCA降维可视化)')
    # plt.subplot(1, 2, 2)
    # plot_decision_boundary(sklearn_lr.predict, X_test_pca, y_test, 'sklearn逻辑回归决策边界\n(PCA降维可视化)')
    # plt.tight_layout()
    # plt.savefig("MyLR.png",dpi=600)
    # plt.show()

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # 1. 加载数据
    data = load_iris()
    X = data.data[:, :2]  # 只使用前两个特征方便可视化（花瓣长度和宽度）
    y = data.target
    feature_names = data.feature_names[:2]
    class_names = data.target_names

    # 2. 标准化 + 划分训练测试集
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 自定义 One-vs-Rest 实现
    models = {}
    for cls in np.unique(y_train):
        y_binary = (y_train == cls).astype(int)
        model = LogisticRegression(lr=1, max_iter=1000, tol=1e-4)
        model.fit(X_train, y_binary, verbose=True)
        models[cls] = model


    def predict_ovr(X):
        probs = np.zeros((X.shape[0], len(models)))
        for cls, model in models.items():
            probs[:, cls] = model.predict_proba(X)
        return np.argmax(probs, axis=1)


    # 4. sklearn 实现
    sklearn_lr =  OneVsRestClassifier(SklearnLogisticRegression(solver='lbfgs', max_iter=1000))
    sklearn_lr.fit(X_train, y_train)

    # 5. 评估模型
    print("自定义逻辑回归:")
    y_pred_custom = predict_ovr(X_test)
    print(classification_report(y_test, y_pred_custom, target_names=class_names))

    print("sklearn逻辑回归:")
    y_pred_sk = sklearn_lr.predict(X_test)
    print(classification_report(y_test, y_pred_sk, target_names=class_names))

    # 6. 可视化设置
    N, M = 500, 500  # 网格采样点数
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    # 合并训练集和测试集用于可视化
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # 7. 自定义模型可视化
    plt.figure(figsize=(15, 6))

    # 自定义模型决策边界
    plt.subplot(1, 2, 1)
    x1_min, x1_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    x2_min, x2_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_test = np.stack((x1.flat, x2.flat), axis=1)

    y_hat = predict_ovr(x_test)
    y_hat = y_hat.reshape(x1.shape)
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light, shading='auto')
    plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, edgecolors='k', s=50, cmap=cm_dark)
    plt.xlabel(feature_names[0] + ' (标准化)')
    plt.ylabel(feature_names[1] + ' (标准化)')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('自定义逻辑回归决策区域')
    plt.grid(True)

    # sklearn模型决策边界
    plt.subplot(1, 2, 2)
    y_hat_sk = sklearn_lr.predict(x_test)
    y_hat_sk = y_hat_sk.reshape(x1.shape)
    plt.pcolormesh(x1, x2, y_hat_sk, cmap=cm_light, shading='auto')
    plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, edgecolors='k', s=50, cmap=cm_dark)
    plt.xlabel(feature_names[0] + ' (标准化)')
    plt.ylabel(feature_names[1] + ' (标准化)')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('sklearn逻辑回归决策区域')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('iris_decision_regions.png', dpi=600)
    # plt.show()

    # 8. 混淆矩阵可视化
    fig_cm, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 自定义逻辑回归混淆矩阵
    cm_custom = confusion_matrix(y_test, y_pred_custom)
    disp_custom = ConfusionMatrixDisplay(confusion_matrix=cm_custom, display_labels=class_names)
    disp_custom.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title('自定义逻辑回归')

    # sklearn逻辑回归混淆矩阵
    cm_sk = confusion_matrix(y_test, y_pred_sk)
    disp_sk = ConfusionMatrixDisplay(confusion_matrix=cm_sk, display_labels=class_names)
    disp_sk.plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title('sklearn 逻辑回归')

    plt.tight_layout()
    plt.savefig("iris_confusion_matrices.png", dpi=600)
    # plt.show()


