import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class BPNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1, activation='sigmoid'):
        # 参数类型检查
        if not all(isinstance(dim, int) and dim > 0 for dim in [input_dim, hidden_dim, output_dim]):
            raise ValueError("input_dim, hidden_dim, output_dim must be positive integers")
        if not isinstance(lr, (float, int)) or lr <= 0:
            raise ValueError("lr must be a positive float")

        # 根据激活函数选择合适的权重初始化方法
        if activation == 'sigmoid':
            # Xavier初始化
            self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim + hidden_dim)
            self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim + output_dim)
        elif activation == 'relu':
            # He初始化
            self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim / 2)
            self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim / 2)
        else:
            raise ValueError("Unsupported activation function. Use 'sigmoid', 'tanh' or 'relu'.")

        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, output_dim))

        # 保存学习率和激活函数选择
        self.lr = lr
        self.activation = activation

    def forward(self, X):
        """前向传播"""
        self._validate_input(X)

        self.z1 = X @ self.W1 + self.b1

        # 根据激活函数类型进行选择
        if self.activation == 'sigmoid':
            self.a1 = sigmoid(self.z1)
        elif self.activation == 'relu':
            self.a1 = relu(self.z1)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """反向传播"""
        self._validate_input(X, y)

        m = X.shape[0]
        dz2 = output - y # 输出层误差
        dW2 = self.a1.T @ dz2 / m # 输出层权重
        db2 = np.sum(dz2, axis=0, keepdims=True) / m # 输出层偏置梯度

        # 根据激活函数类型选择对应的导数计算，回传误差进隐藏层
        if self.activation == 'sigmoid':
            dz1 = dz2 @ self.W2.T * sigmoid_deriv(self.z1)
        elif self.activation == 'relu':
            dz1 = dz2 @ self.W2.T * relu_deriv(self.z1)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        # 隐藏层梯度
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 更新权重和偏置
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        """训练函数"""
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        self._validate_input(X, y)

        for i in range(epochs):
            out = self.forward(X)
            loss = np.mean((y - out) ** 2)
            self.backward(X, y, out)
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        """预测函数"""
        self._validate_input(X)
        probs = self.forward(X)
        return (probs > 0.5).astype(int)

    def accuracy(self, X, y):
        """准确率计算"""
        self._validate_input(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred.flatten() == y.flatten())

    def _validate_input(self, X, y=None):
        """检查输入数据和标签的形状和类型"""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array")
            if y.shape[0] != X.shape[0]:
                raise ValueError(f"X and y batch size mismatch: {X.shape[0]} vs {y.shape[0]}")
            if y.ndim != 2 or y.shape[1] != self.b2.shape[1]:
                raise ValueError(f"y shape must be (batch_size, {self.b2.shape[1]}), got {y.shape}")


if __name__ == "__main__":
    # 数据生成
    np.random.seed(42)
    num_samples = 1000
    noise = 0.3

    # 生成类别1（内圆）的数据
    radius_inner = 1.0
    angles_inner = 2 * np.pi * np.random.rand(num_samples)
    x1_inner = radius_inner * np.cos(angles_inner) + np.random.normal(0, noise, num_samples)
    y1_inner = radius_inner * np.sin(angles_inner) + np.random.normal(0, noise, num_samples)
    X1_inner = np.vstack((x1_inner, y1_inner)).T

    # 生成类别2（外圆）的数据
    radius_outer = 2.0
    angles_outer = 2 * np.pi * np.random.rand(num_samples)
    x2_outer = radius_outer * np.cos(angles_outer) + np.random.normal(0, noise, num_samples)
    y2_outer = radius_outer * np.sin(angles_outer) + np.random.normal(0, noise, num_samples)
    X2_outer = np.vstack((x2_outer, y2_outer)).T

    # 合并数据
    X = np.vstack((X1_inner, X2_outer))
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y = np.hstack((np.ones(num_samples), np.zeros(num_samples)))  # 类别1为1，类别2为0
    y = y.reshape(-1, 1)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    # Initialize the MLPClassifier (single hidden layer)
    mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', solver='adam', learning_rate_init=0.1,
                        max_iter=1000)

    # Train the model
    mlp.fit(X_train, y_train.ravel())

    # Make predictions
    y_pred_mlp = mlp.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_mlp)
    print(f"Accuracy of the MLPClassifier: {accuracy:.4f}")

    # 可视化数据
    plt.figure(figsize=(8, 8))
    plt.scatter(X1_inner[:, 0], X1_inner[:, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
    plt.scatter(X2_outer[:, 0], X2_outer[:, 1], c='red', label='类别2', alpha=0.6, edgecolors='k')
    plt.title('非线性可分数据集：内外圆')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # BP神经网络模型
    input_dim = 2  # 输入维度：2（每个数据点有2个特征）
    hidden_dim = 5  # 隐藏层维度：可以根据实验调整
    output_dim = 1  # 输出维度：1（二分类）

    # 实例化神经网络
    # 使用 ReLU 激活函数
    bpnn = BPNeuralNetwork(input_dim, hidden_dim, output_dim, lr=0.2, activation='relu')

    # 训练模型
    bpnn.train(X_train, y_train, epochs=2000)

    # 测试模型
    train_accuracy = bpnn.accuracy(X_train, y_train)
    test_accuracy = bpnn.accuracy(X_test, y_test)

    print(f"训练集准确度: {train_accuracy:.4f}")
    print(f"测试集准确度: {test_accuracy:.4f}")

    # 预测
    y_pred_bp = bpnn.predict(X_test).flatten()
    y_true = y_test.flatten()

    # 可视化比较图（1x2），红圈标出预测错误的点
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # MLP结果
    axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_mlp, cmap='coolwarm', alpha=0.6, edgecolors='k')
    wrong_idx_mlp = (y_pred_mlp != y_true)
    axes[0].scatter(X_test[wrong_idx_mlp, 0], X_test[wrong_idx_mlp, 1], facecolors='none', edgecolors='red', s=100, label='错误预测')
    axes[0].set_title('MLPClassifier 测试集预测结果')
    axes[0].set_xlabel('特征 1')
    axes[0].set_ylabel('特征 2')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axis('equal')

    # BP神经网络结果
    axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_bp, cmap='coolwarm', alpha=0.6, edgecolors='k')
    wrong_idx_bp = (y_pred_bp != y_true)
    axes[1].scatter(X_test[wrong_idx_bp, 0], X_test[wrong_idx_bp, 1], facecolors='none', edgecolors='red', s=100, label='错误预测')
    axes[1].set_title('BP神经网络 测试集预测结果')
    axes[1].set_xlabel('特征 1')
    axes[1].set_ylabel('特征 2')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].axis('equal')

    plt.tight_layout()
    plt.savefig("BP对比图.png", dpi=600)
    plt.show()

