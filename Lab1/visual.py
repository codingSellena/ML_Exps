import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import seaborn as sns


def plot_decision_boundary(X, y, model):
    # X 是二维数据集，y 是标签，model 是训练好的分类模型

    # 创建一个网格，遍历所有可能的特征值
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 获取预测结果
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu, s=50)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def draw_confusion_matrix(lr, X_test, y_test):
    # 假设 y_true 是真实标签，y_pred 是模型预测结果
    y_pred = lr.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # 使用 seaborn 绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def draw_roc_curve(lr, X_test, y_test):
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test))
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def draw_precision_recall_curve(lr, X_test, y_test):
    # 计算 Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(y_test, lr.predict_proba(X_test))

    # 绘制 Precision-Recall 曲线
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
