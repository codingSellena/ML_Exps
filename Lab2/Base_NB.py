import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

class NaiveBayes:
    def __init__(self, model_type='gaussian', alpha=1.0, binarize_threshold=0.0):
        """
        model_type: 'gaussian', 'multinomial', or 'bernoulli'
        alpha: 拉普拉斯平滑参数
        binarize_threshold: 伯努利模型的二值化阈值
        """
        self.model_type = model_type
        self.alpha = alpha
        self.binarize_threshold = binarize_threshold
        self.classes = None
        self.class_priors = None
        self.parameters = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples = X.shape[0]

        # 计算先验概率
        self.class_priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = (np.sum(y == c) + self.alpha) / (n_samples + n_classes * self.alpha)

        # 根据模型类型计算条件概率参数
        if self.model_type == 'gaussian':
            for i, c in enumerate(self.classes):
                X_c = X[y == c]
                self.parameters[c] = {
                    'mean': X_c.mean(axis=0),
                    'var': X_c.var(axis=0) + 1e-9
                }
        elif self.model_type == 'multinomial':
            # \theta_{cj} = (类c中所有样本在特征j上的总和+alpha)/(类c中所有特征的总和+\alpha*D)
            for i, c in enumerate(self.classes):
                X_c = X[y == c]
                total_count = np.asarray(X_c.sum(axis=0)).flatten() + self.alpha
                self.parameters[c] = {
                    'theta': total_count / (total_count.sum() + X_c.shape[1] * self.alpha)
                }
        elif self.model_type == 'bernoulli':
            # p_{cj} = (类c中特征j出现的样本数+alpha)/(类c中的样本数+\alpha*2)
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = (X > self.binarize_threshold).astype(int)
            for i, c in enumerate(self.classes):
                X_c = X[y == c]
                self.parameters[c] = {
                    'p': (X_c.sum(axis=0) + self.alpha) / (X_c.shape[0] + 2 * self.alpha)
                }

    def _calculate_log_likelihood(self, x, c):
        if self.model_type == 'gaussian':
            mean = self.parameters[c]['mean']
            var = self.parameters[c]['var']
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var + 1e-9)) \
                             - 0.5 * np.sum((x - mean) ** 2 / var)
            return log_likelihood
        elif self.model_type == 'multinomial':
            theta = self.parameters[c]['theta'].flatten()
            x = x.flatten()
            return np.dot(x.T, np.log(theta + 1e-9))
        elif self.model_type == 'bernoulli':
            x_bin = (x > self.binarize_threshold).astype(int).flatten()
            p = self.parameters[c]['p'].flatten()
            log_p = np.log(np.clip(p, 1e-9, 1))
            log_not_p = np.log(np.clip(1 - p, 1e-9, 1))
            return np.dot(x_bin, log_p) + np.dot(1 - x_bin, log_not_p)
        raise TypeError(f'{self.model_type} is not a valid model type')

    def predict(self, X):
        """预测"""
        # 转换输入格式
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 预分配结果数组
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for i, c in enumerate(self.classes):
            # logP(y=c) + \sum_{j=1} log P(x_j|y=c)
            log_prior = np.log(self.class_priors[i])
            log_probs[:, i] = log_prior + np.array([
                self._calculate_log_likelihood(x, c) for x in X
            ])

        return self.classes[np.argmax(log_probs, axis=1)]

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

    # 1. Iris 数据集 (GaussianNB)
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3,
                                                                            random_state=42)

    # 使用自己实现的 GaussianNB
    nb_iris = NaiveBayes(model_type='gaussian')
    nb_iris.fit(X_train_iris, y_train_iris)
    y_pred_iris = nb_iris.predict(X_test_iris)
    print("Iris Data - Custom GaussianNB Accuracy: ", accuracy_score(y_test_iris, y_pred_iris))

    # 使用 sklearn 的 GaussianNB
    gnb_sklearn = GaussianNB()
    gnb_sklearn.fit(X_train_iris, y_train_iris)
    y_pred_iris_sklearn = gnb_sklearn.predict(X_test_iris)
    print("Iris Data - Sklearn GaussianNB Accuracy: ", accuracy_score(y_test_iris, y_pred_iris_sklearn))


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test_iris, y_pred_iris), display_labels=iris.target_names).plot(ax=axes[0], colorbar=False,cmap="Blues")
    axes[0].set_title("Custom GaussianNB")

    ConfusionMatrixDisplay(confusion_matrix(y_test_iris, y_pred_iris_sklearn), display_labels=iris.target_names).plot(ax=axes[1], colorbar=False,cmap="Blues")
    axes[1].set_title("Sklearn GaussianNB")

    plt.tight_layout()
    plt.savefig("confusion_iris_gaussian.png")

    # 2. Breast Cancer 数据集
    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(X_cancer, y_cancer, test_size=0.3,
                                                                                    random_state=42)

    # 使用自己实现的 MultinomialNB
    nb_cancer = NaiveBayes(model_type='multinomial', alpha=1.0)
    nb_cancer.fit(X_train_cancer, y_train_cancer)
    y_pred_cancer = nb_cancer.predict(X_test_cancer)
    print("Breast Cancer Data - Custom MultinomialNB Accuracy: ", accuracy_score(y_test_cancer, y_pred_cancer))

    # 使用 sklearn 的 MultinomialNB
    bnb_sklearn = MultinomialNB(alpha=1.0)
    bnb_sklearn.fit(X_train_cancer, y_train_cancer)
    y_pred_cancer_sklearn = bnb_sklearn.predict(X_test_cancer)
    print("Breast Cancer Data - Sklearn MultinomialNB Accuracy: ", accuracy_score(y_test_cancer, y_pred_cancer_sklearn))

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test_cancer, y_pred_cancer),
                           display_labels=cancer.target_names).plot(ax=axes[0], colorbar=False,cmap="Blues")
    axes[0].set_title("Custom MultinomialNB")

    ConfusionMatrixDisplay(confusion_matrix(y_test_cancer, y_pred_cancer_sklearn),
                           display_labels=cancer.target_names).plot(ax=axes[1], colorbar=False,cmap="Blues")
    axes[1].set_title("Sklearn MultinomialNB")

    plt.tight_layout()
    plt.savefig("confusion_cancer_multinomial.png")

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    # 2. 20 Newsgroups 数据集
    categories = ['rec.sport.baseball', 'sci.med']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories)
    vectorizer = CountVectorizer(binary=True, stop_words='english', max_features=1000)
    X_news = vectorizer.fit_transform(newsgroups.data)
    y_news = newsgroups.target

    X_train_news, X_test_news, y_train_news, y_test_news = train_test_split(X_news, y_news, test_size=0.3,
                                                                            random_state=42)

    # 使用自己实现的 BernoulliNB
    nb_news = NaiveBayes(model_type='bernoulli', alpha=1.0)
    nb_news.fit(X_train_news, y_train_news)
    y_pred_news = nb_news.predict(X_test_news)
    print("20News - Custom BernoulliNB Accuracy: ", accuracy_score(y_test_news, y_pred_news))

    # 使用 sklearn 的 BernoulliNB
    bnb_sklearn = BernoulliNB(alpha=1.0)
    bnb_sklearn.fit(X_train_news, y_train_news)
    y_pred_news_sklearn = bnb_sklearn.predict(X_test_news)
    print("20News - Sklearn BernoulliNB Accuracy: ", accuracy_score(y_test_news, y_pred_news_sklearn))

    # 可视化混淆矩阵
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test_news, y_pred_news),
                           display_labels=categories).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Custom BernoulliNB")

    ConfusionMatrixDisplay(confusion_matrix(y_test_news, y_pred_news_sklearn),
                           display_labels=categories).plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Sklearn BernoulliNB")

    plt.tight_layout()
    plt.savefig("confusion_news_bernoulli.png")
