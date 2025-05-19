import numpy as np
from collections import deque,Counter


class KDTreeNode:
    def __init__(self, point=None, index=None, left=None, right=None,axis=None):
        self.point = point  # 数据点
        self.index = index  # 在训练集中的原始索引
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.axis = axis  # 分割维度


class KDTree:
    def __init__(self, data):
        """
        :param data: 训练数据 (n_samples, n_features)
        """
        self.n_features = data.shape[1]
        self.root = self._build_tree(data, depth=0)

    def _build_tree(self, data, depth, indices=None):
        if len(data) == 0:
            return None

        if indices is None:
            indices = np.arange(len(data))

        axis = depth % self.n_features
        sorted_indices = np.argsort(data[:, axis])
        median_idx = len(sorted_indices) // 2

        node = KDTreeNode(
            point=data[sorted_indices[median_idx]],
            index=indices[sorted_indices[median_idx]],
            axis=axis
        )

        # 递归构建子树
        left_indices = indices[sorted_indices[:median_idx]]
        right_indices = indices[sorted_indices[median_idx + 1:]]

        node.left = self._build_tree(
            data[sorted_indices[:median_idx]],
            depth + 1,
            left_indices
        )
        node.right = self._build_tree(
            data[sorted_indices[median_idx + 1:]],
            depth + 1,
            right_indices
        )

        return node

    def query(self, target, k=1):
        """
        查询k近邻
        :param target: 查询点 (n_features,)
        :param k: 需要返回的邻居数量
        :return: (indices, distances)
        """
        if not hasattr(target, '__len__') or len(target) != self.n_features:
            raise ValueError("目标点维度不匹配")

        # 使用最大堆保存结果
        neighbors = []
        self._search(self.root, target, k, neighbors)

        # 按距离排序并返回
        neighbors.sort(key=lambda x: x[1])
        indices = [n[0] for n in neighbors[:k]]
        distances = [n[1] for n in neighbors[:k]]
        return np.array(indices), np.array(distances)

    def _search(self, node, target, k, neighbors, depth=0):
        if node is None:
            return

        # 计算当前节点距离
        dist = np.linalg.norm(node.point - target)

        # 维护大小为k的堆
        if len(neighbors) < k:
            neighbors.append((node.index, dist))
        elif dist < neighbors[-1][1]:
            neighbors[-1] = (node.index, dist)
        neighbors.sort(key=lambda x: x[1])  # 保持有序

        # 决定搜索路径
        axis = node.axis
        if target[axis] < node.point[axis]:
            self._search(node.left, target, k, neighbors, depth + 1)
        else:
            self._search(node.right, target, k, neighbors, depth + 1)

        # 检查另一侧子树
        if len(neighbors) < k or abs(target[axis] - node.point[axis]) < neighbors[-1][1]:
            other_node = node.right if target[axis] < node.point[axis] else node.left
            self._search(other_node, target, k, neighbors, depth + 1)

class KNN:
    def __init__(self, k=3, distance_metric='euclidean', use_kdtree=False):
        """
        初始化KNN分类器
        :param k: 邻居数量
        :param distance_metric: 距离度量
        :param use_kdtree: 是否使用KD-Tree加速搜索
        """
        self.k = k
        self.distance_metric = distance_metric
        self.use_kdtree = use_kdtree
        self.X_train = None
        self.y_train = None
        self.kdtree = None

    def fit(self, X, y):
        """
        训练模型（KNN只需存储训练数据）
        :param X: 训练特征 (n_samples, n_features)
        :param y: 训练标签 (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        if self.use_kdtree:
            # 使用KD-Tree构建树
            self.kdtree = KDTree(self.X_train)

    def predict(self, X):
        """
        预测样本类别
        :param X: 测试样本 (n_samples, n_features)
        :return: 预测标签 (n_samples,)
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        预测单个样本
        """
        if self.use_kdtree:
            # 使用KD-Tree加速搜索
            neighbors = self._get_nearest_neighbors_kdtree(x)
        else:
            # 使用暴力搜索（欧几里得或曼哈顿距离）
            neighbors = self._get_nearest_neighbors(x)

        # 获取最近的k个样本的标签
        k_nearest_labels = [self.y_train[i] for i in neighbors]

        # 投票决定预测结果
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _get_nearest_neighbors(self, x):
        """
        获取最近的k个邻居
        """
        distances = []
        if self.distance_metric == "euclidean":
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == "manhattan":
            distances = [self._manhattan_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        return k_indices

    def _get_nearest_neighbors_kdtree(self, x):
        """
        使用KD-Tree获取k近邻
        """
        indices, _ = self.kdtree.query(x, k=self.k)
        return indices

    def _euclidean_distance(self, x1, x2):
        """
        欧氏距离计算
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        """
        曼哈顿距离计算
        """
        return np.sum(np.abs(x1 - x2))

    def score(self, X, y):
        """
        计算模型准确率
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

if __name__ == '__main__':
    import time
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # ---------- 加载与预处理数据 ----------
    data = load_breast_cancer()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------- 精度随 K 值变化 ----------
    ks = range(1, 21)
    my_scores = []
    sk_scores = []
    brute_scores = []
    my_predict_times = []
    sk_predict_times = []
    brute_predict_times = []

    for k in ks:
        # 自定义KNN
        my_knn = KNN(k=k, use_kdtree=True)
        my_knn.fit(X_train, y_train)

        # 测量预测时间
        start_time = time.time()
        my_preds = my_knn.predict(X_test)
        my_predict_times.append(time.time() - start_time)

        my_scores.append(accuracy_score(y_test, my_preds))

        # 自定义KNN(brute)
        my_knn_brute = KNN(k=k,distance_metric="euclidean", use_kdtree=False)
        my_knn_brute.fit(X_train, y_train)

        # 测量预测时间
        start_time = time.time()
        my_preds_brute = my_knn_brute.predict(X_test)
        brute_predict_times.append(time.time() - start_time)

        # sklearn KNN (使用KDTree)
        sk_knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
        sk_knn.fit(X_train, y_train)

        # 测量预测时间
        start_time = time.time()
        sk_preds = sk_knn.predict(X_test)
        sk_predict_times.append(time.time() - start_time)

        sk_scores.append(accuracy_score(y_test, sk_preds))

    # ---------- 绘图 ----------
    plt.figure(figsize=(10, 6))

    # 精度图
    plt.subplot(2, 1, 1)
    plt.plot(ks, my_scores, label='My KNN (KDTree)', marker='o')
    plt.plot(ks, sk_scores, label='Sklearn KNN (KDTree)', marker='x')
    plt.xlabel("K 值")
    plt.ylabel("准确率")
    plt.title("K 值对准确率的影响（乳腺癌数据集）")
    plt.xticks(ks)
    plt.grid(True)
    plt.legend()

    # 时间图
    plt.subplot(2, 1, 2)
    plt.plot(ks, my_predict_times, label='My KNN Predict Time (KDTree)', marker='o')
    plt.plot(ks, sk_predict_times, label='Sklearn KNN Predict Time (KDTree)', marker='x')
    plt.plot(ks, brute_predict_times, label='My KNN Predict Time (Brute)', marker='s')
    plt.xlabel("K 值")
    plt.ylabel("时间 (秒)")
    plt.title("K 值对训练与预测时间的影响")
    plt.xticks(ks)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("knn_comparison_with_brute_euclidean.png", dpi=600)
    plt.show()


    #from sklearn.datasets import load_iris
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import accuracy_score
    #
    # # 加载数据
    # iris = load_iris()
    # X, y = iris.data, iris.target
    #
    # # 划分训练测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # 创建KNN分类器
    # knn = KNN(k=5, distance_metric='euclidean')
    #
    # # 训练模型
    # knn.fit(X_train, y_train)
    #
    # # 预测
    # predictions = knn.predict(X_test)
    #
    # # 评估
    # accuracy = knn.score(X_test, y_test)
    # print(f"测试集准确率: {accuracy:.2f}")
    #
    # # 打印示例预测结果
    # print("\n前5个测试样本的预测结果:")
    # for i in range(5):
    #     print(f"真实标签: {y_test[i]}, 预测标签: {predictions[i]}")