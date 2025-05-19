import numpy as np

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.priors = {}  # P(y)
        self.means = {}  # mean of feature per class
        self.vars = {}  # variance of feature per class

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0] # P(y=c)
            self.means[c] = X_c.mean(axis=0)
            self.vars[c] = X_c.var(axis=0) + 1e-9  # avoid divide by zero

    def _gaussian_pdf(self, class_idx, x):
        """P(x_i | y=c) = 1/\sqrt{2*pi*\sigma^2}e^(-\frac{(xi-\mu)^2}{2\sigma^2})"""
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_instance(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            # log P(x | y=c) = \sum log Gaussian(x_i | \mu_c[i], \sigma^2_c[i])
            class_conditional = np.sum(np.log(self._gaussian_pdf(c, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_instance(x) for x in X])


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
    from sklearn.metrics import accuracy_score,classification_report
    # 加载 breast_cancer 数据集
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

    # 自定义模型
    GNB = GaussianNB()
    GNB.fit(X_train, y_train)
    custom_pred = GNB.predict(X_test)
    custom_acc = accuracy_score(y_test, custom_pred)
    print("My Accuracy:", accuracy_score(y_test, custom_pred))
    print(classification_report(y_test, custom_pred, target_names=data.target_names))

    # sklearn 模型
    sk_nb = SklearnGaussianNB()
    sk_nb.fit(X_train, y_train)
    sk_pred = sk_nb.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_pred)
    print("Sklearn Accuracy:", accuracy_score(y_test, sk_pred))
    print(classification_report(y_test, sk_pred, target_names=data.target_names))

    # from sklearn.datasets import load_iris
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics import accuracy_score, classification_report
    #
    # # 加载数据
    # data = load_iris()
    # X = data.data
    # y = data.target
    #
    # # 数据预处理
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #
    # # 训练和预测
    # gnb = GaussianNB()
    # gnb.fit(X_train, y_train)
    # y_pred = gnb.predict(X_test)
    #
    # # 评估
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred, target_names=data.target_names))
