import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from math import log
from graphviz import Digraph


class DecisionTreeID3:
    def __init__(self, max_depth=None, min_samples_split=2,
                 pre_pruning=False, min_info_gain=1e-6):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.pre_pruning = pre_pruning
        self.min_info_gain = min_info_gain
        self.tree = None
        self.feature_names = None
        self.label_name = None

    def fit(self, X, y, feature_names=None, label_name="label"):
        """
        训练决策树
        :param X: 特征数据 (List[List] 或 pd.DataFrame)
        :param y: 标签数据 (List 或 pd.Series)
        :param feature_names: 特征名称列表
        :param label_name: 标签名称
        """
        # 数据预处理
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values.tolist()
        if isinstance(y, pd.Series):
            y = y.tolist()

        # 组合特征和标签
        D = [x + [y[i]] for i, x in enumerate(X)]

        self.feature_names = feature_names or [f"feature_{i}" for i in range(len(X[0]))]
        self.label_name = label_name
        self.tree = self._build_tree(D, set(range(len(X[0]))), depth=0)

    def _build_tree(self, D, A, depth):
        labels = [sample[-1] for sample in D] # 获取标签

        # 递归停止条件
        if (len(set(labels)) == 1 or
                len(D) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                not A):
            return self._most_common_label(labels)

        best_attr = max(A, key=lambda a: self._information_gain(D, a))
        info_gain = self._information_gain(D, best_attr)

        # 预剪枝判断
        if self.pre_pruning and info_gain < self.min_info_gain:
            return self._most_common_label(labels)

        node = {
            'attribute': best_attr,
            'name': self.feature_names[best_attr],
            'branches': {},
            'samples':len(D)
        }

        values = set(sample[best_attr] for sample in D)
        for val in values:
            subset = [sample for sample in D if sample[best_attr] == val]
            node['branches'][val] = self._build_tree(subset, A - {best_attr}, depth + 1)

        return node

    def predict(self, X):
        """ 预测 """
        if isinstance(X, pd.DataFrame):
            X = X.values.tolist()
        return [self._predict_sample(self.tree, x) for x in X]

    def _predict_sample(self, node, sample):
        if isinstance(node, dict) and 'label' in node: # 判断叶子节点
            return node['label']

        attr_val = sample[node['attribute']] # 提取特征值
        if attr_val not in node['branches']: # 若为不存在的特征，返回出现最多的类别
            return self._most_common_label_from_node(node)

        return self._predict_sample(node['branches'][attr_val], sample)

    def visualize(self, filename='decision_tree', feature_names=None, class_names=None, discretizer=None):
        """ 决策树可视化 """
        dot = Digraph(
            name='Decision Tree',
            node_attr={'shape': 'box', 'style': 'rounded,filled', 'fontname': 'Helvetica'},
            graph_attr={'dpi': '300'}
        )

        # 自动获取类别名称
        if class_names is None and hasattr(self, 'classes_'):
            class_names = self.classes_

        self._add_sklearn_nodes(
            dot,
            self.tree,
            parent_id=None,
            edge_label="",
            feature_names=feature_names or self.feature_names,
            class_names=class_names,
            discretizer=discretizer
        )
        dot.render(filename, view=True, format='png', cleanup=True)
        return dot

    def _add_sklearn_nodes(self, dot, node, parent_id, edge_label, feature_names, class_names, discretizer=None):
        """ 添加节点 """
        if isinstance(node, dict) and 'label' not in node:
            # 获取节点信息
            attr_index = node['attribute']
            attr_name = feature_names[attr_index]
            samples = self._count_samples(node)

            # 如果提供了 discretizer，获取分箱的边界
            bin_edges = None
            if discretizer:
                bin_edges = discretizer.bin_edges_[attr_index]
                bin_label = f"Bins: {bin_edges}"
            else:
                bin_label = "No Binning"

            # 创建节点标签（类似sklearn的格式）
            node_label = (
                f"<font face='Helvetica'><b>{attr_name}</b><br/>"
                f"samples = {samples}<br/>"
                f"value = {self._format_value(node, class_names)}<br/>"
                f"class = {self._get_node_class(node, class_names)}<br/>"
                f"{bin_label}</font>"
            )

            # 创建节点
            node_id = str(id(node))
            dot.node(
                name=node_id,
                label=f"<{node_label}>",
                fillcolor=self._get_node_color(node)
            )

            # 添加连接线
            if parent_id:
                dot.edge(parent_id, node_id, label=str(edge_label))

            # 递归添加子节点
            for val, child in node['branches'].items():
                self._add_sklearn_nodes(
                    dot, child, node_id, val,
                    feature_names, class_names,
                    discretizer
                )
        else:
            # 叶子节点
            leaf_label = (
                f"<font face='Helvetica'><b>class</b><br/>"
                f"samples = {self._count_samples(node)}<br/>"
                f"value = {self._format_leaf_value(node, class_names)}<br/>"
                f"class = {class_names[node.get('label', None)] if class_names is not None else node}</font>"
            )

            leaf_id = f"leaf_{id(node)}"
            dot.node(
                name=leaf_id,
                label=f"<{leaf_label}>",
                fillcolor=self._get_leaf_color(node)
            )
            dot.edge(parent_id, leaf_id, label=str(edge_label))

    def _count_samples(self, node):
        if isinstance(node, dict):
            return node.get('samples', sum(self._count_samples(child) for child in node.get('branches', {}).values()))
        return node.get('samples', 1)  # 返回结点样本数

    def _format_value(self, node, class_names):
        """ 格式化显示每个类别的实际样本数 """
        counts = self._get_label_counts(node)
        if class_names is not None:
            # 返回形式: [类别0样本数, 类别1样本数, ...]
            return [counts.get(i, 0) for i in range(len(class_names))]
        # 如果没有类别名称，返回所有类别的样本数列表
        return list(counts.values())

    def _format_leaf_value(self, node, class_names):
        """ 格式化叶子节点的样本显示 """
        if class_names is not None:
            # 叶子节点只有一个类别，其他类别为0
            label = node.get('label', 0)
            return [node.get('samples', 1) if i == label else 0 for i in range(len(class_names))]
        # 如果没有类别名称，返回总样本数
        return [node.get('samples', 1)]

    def _get_label_counts(self, node):
        """ 获取节点类别统计（包含样本量） """
        counts = Counter()
        if isinstance(node, dict):
            if 'label' in node:  # 叶子节点
                counts[node['label']] = node.get('samples', 1)
            else:  # 分裂节点
                for child in node['branches'].values():
                    counts.update(self._get_label_counts(child))
        return counts

    def _get_node_class(self, node, class_names):
        """ 获取节点主要类别 """
        counts = self._get_label_counts(node)
        majority = counts.most_common(1)[0][0]
        return class_names[majority] if class_names is not None else majority

    def _get_node_color(self, node):
        """ 节点颜色 """
        return "#FFF2CC"  # 浅黄色

    def _get_leaf_color(self, node):
        """ 叶子节点颜色 """
        colors = {
            0: "#FFCCCC",  # 红色
            1: "#CCE5FF",  # 蓝色
            2: "#CCFFCC"  # 绿色
        }

        return colors.get(node.get('label', 0) % len(colors), "#FFFFFF")

    def _most_common_label(self, labels):
        label = Counter(labels).most_common(1)[0][0]
        return {'label': label, 'samples': len(labels), 'is_leaf': True}

    def _most_common_label_from_node(self, node):
        """ 从节点中提取最常见的标签 """
        if isinstance(node, dict) and 'label' in node:
            return node

        labels = []
        for branch in node['branches'].values():
            if isinstance(branch, dict):
                labels.extend(self._get_all_labels(branch))
            else:
                labels.append(branch)
        label = Counter(labels).most_common(1)[0][0]
        return {'label': label, 'samples': len(labels), 'is_leaf': True}

    def _get_all_labels(self, node):
        """ 递归获取节点下所有标签 """
        labels = []
        if isinstance(node, dict) and 'label' in node:
            labels.append(node['label'])
        elif isinstance(node, dict):
            for branch in node['branches'].values():
                labels.extend(self._get_all_labels(branch))
        else:
            labels.append(node)
        return labels

    def _entropy(self, D):
        # E(D) = - \sum p_i log_2 p_i
        # p_i = count(i) / len(labels)
        labels = [sample[-1] for sample in D]
        total = len(labels)
        counter = Counter(labels)
        return -sum((count / total) * log((count / total) + 1e-9, 2) for count in counter.values())

    def _information_gain(self, D, attr_index):
        # E(D|A) = \sum D_i/D * E(D_i)
        # Gain(D,A) = E(D) - E(D|A)
        total_entropy = self._entropy(D)
        values = set(sample[attr_index] for sample in D)
        cond_entropy = 0.0
        for val in values:
            subset = [sample for sample in D if sample[attr_index] == val]
            prob = len(subset) / len(D)
            cond_entropy += prob * self._entropy(subset)
        return total_entropy - cond_entropy

    def post_prune(self, X_val, y_val):
        """
        后剪枝
        """
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values.tolist()
        if isinstance(y_val, pd.Series):
            y_val = y_val.tolist()

        # 创建验证数据集
        D_val = [x + [y_val[i]] for i, x in enumerate(X_val)]

        # 执行剪枝
        self.tree = self._prune_node(self.tree, D_val)

    def _prune_node(self, node, D_val):
        # 如果是叶子节点，直接返回
        if isinstance(node, dict) and 'label' in node:
            return node

        # 递归剪枝子树
        for val in list(node['branches'].keys()):
            # 获取当前分支对应的验证数据子集,其他验证数据不影响该部分准确度的计算
            subset = [sample for sample in D_val if sample[node['attribute']] == val]
            if subset:  # 只对存在的分支剪枝
                node['branches'][val] = self._prune_node(node['branches'][val], subset)

        # 计算当前节点的验证集准确率
        original_acc = self._node_accuracy(node, D_val)

        # 创建剪枝后的节点（多数类叶子）
        pruned_node = self._most_common_label([sample[-1] for sample in D_val])

        # 计算剪枝后的准确率
        pruned_acc = self._leaf_accuracy(pruned_node['label'], D_val)

        # 决定是否剪枝
        if pruned_acc >= original_acc:
            return pruned_node
        return node

    def _node_accuracy(self, node, D):
        """计算节点在数据集D上的准确率"""
        correct = 0
        for sample in D:
            pred = self._predict_sample(node, sample[:-1])
            if pred == sample[-1]:
                correct += 1
        return correct / len(D) if D else 0

    def _leaf_accuracy(self, label, D):
        """计算叶子节点在数据集D上的准确率"""
        if not D:
            return 0
        return sum(1 for sample in D if sample[-1] == label) / len(D)



from sklearn.preprocessing import KBinsDiscretizer

def discretize_data(X, n_bins=4, strategy='quantile'):
    """
    使用KBinsDiscretizer将连续特征离散化
    :param X: np.ndarray 类型的特征数据
    :param n_bins: 离散化后的桶数
    :param strategy: 'uniform'（均匀间隔）、'quantile'（分位数）、'kmeans'
    :return: 离散化后的X（List[List]），和离散器
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    X_binned = discretizer.fit_transform(X)
    return X_binned.tolist(), discretizer


# 示例使用
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # 加载数据集
    data = load_breast_cancer()
    X, y = data.data, data.target

    # feature_values = [np.unique(X[:, i]) for i in range(X.shape[1])]

    feature_names = data.feature_names.tolist()
    # for i, vals in enumerate(feature_values):
    #     print(f"特征 {feature_names[i]} 的取值有：{vals}")
    # 离散化数据
    X_discrete, discretizer = discretize_data(X, n_bins=2, strategy='quantile')

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, test_size=0.2, random_state=42)

    # 使用你的 ID3 决策树
    model = DecisionTreeID3(max_depth=5,pre_pruning=False)
    model.fit(X_train, y_train, feature_names=feature_names, label_name="label")
    #model.post_prune(X_test, y_test)
    y_pred = model.predict(X_test)
    y_pred = [item['label'] if isinstance(item, dict) else int(item) for item in y_pred]
    # print(feature_names)
    # print(data.target_names)
    model.visualize(filename='2bin_id3tree_without_post_pruning',
                    feature_names=feature_names,
                    class_names=data.target_names,
                    discretizer=discretizer)

    from sklearn.metrics import accuracy_score, confusion_matrix

    print(f"自定义 ID3 准确率: {accuracy_score(y_test, y_pred):.4f}")

    from sklearn.tree import DecisionTreeClassifier, export_graphviz

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    clf.fit(X_train, y_train)
    y_pred_sklearn = clf.predict(X_test)
    print(f"sklearn ID3 准确率: {accuracy_score(y_test, y_pred_sklearn):.4f}")

    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=feature_names,
                               class_names=data.target_names,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("sktree_without_post", view=True, format='png')  # 输出为 tree.pdf
    # graph.view()

    # 混淆矩阵计算
    cm_custom = confusion_matrix(y_test, y_pred)
    cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)

    # 绘制混淆矩阵对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 自定义 ID3 混淆矩阵
    axes[0].imshow(cm_custom, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Custom ID3 Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    tick_marks = np.arange(len(data.target_names))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(data.target_names)
    axes[0].set_yticklabels(data.target_names)
    for i in range(cm_custom.shape[0]):
        for j in range(cm_custom.shape[1]):
            axes[0].text(j, i, format(cm_custom[i, j]), ha="center", va="center", color="white")

    # sklearn ID3 混淆矩阵
    axes[1].imshow(cm_sklearn, interpolation='nearest', cmap='Blues')
    axes[1].set_title('Sklearn ID3 Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(data.target_names)
    axes[1].set_yticklabels(data.target_names)
    for i in range(cm_sklearn.shape[0]):
        for j in range(cm_sklearn.shape[1]):
            axes[1].text(j, i, format(cm_sklearn[i, j]), ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig("ID3_confusion_matrix_with_post.png",dpi=600)
    # plt.show()


    # import pandas as pd
    # from sklearn.preprocessing import LabelEncoder
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.metrics import accuracy_score
    #
    # # 1. 读取数据
    # df = pd.read_csv("data_word.csv")  # 西瓜数据集
    # X_raw = df.iloc[:, :-1]
    # y_raw = df.iloc[:, -1]
    #
    # # 2. 编码类别数据
    # encoders = {}
    # X_encoded = pd.DataFrame()
    # for col in X_raw.columns:
    #     le = LabelEncoder()
    #     X_encoded[col] = le.fit_transform(X_raw[col])
    #     encoders[col] = le  # 保存编码器以便转换测试样本
    #
    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(y_raw)
    #
    # X_list = X_raw.values.tolist()
    # y_list = y_encoded.tolist()
    # model = DecisionTreeID3(max_depth=3)
    # model.fit(X_list, y_list, feature_names=X_raw.columns.tolist(), label_name="好瓜")
    #
    # # 4. sklearn ID3（设置 criterion='entropy'）
    # clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    # clf.fit(X_encoded.values, y_encoded)
    #
    # # 5. 测试样本
    # test_data_raw = ['乌黑', '稍蜷', '浊响', '清晰', '凹陷', '硬滑']
    # test_data_df = pd.DataFrame([test_data_raw], columns=X_raw.columns)
    # test_data_encoded = [
    #     encoders[col].transform([val])[0] for col, val in zip(X_raw.columns, test_data_raw)
    # ]
    #
    # # 自定义模型预测（用原始值）
    # print(f"[自定义ID3] 预测结果: {'好瓜' if model.predict([test_data_raw])[0] == 1 else '坏瓜'}")
    #
    # # sklearn 模型预测（用编码值）
    # pred = clf.predict([test_data_encoded])[0]
    # print(f"[sklearn ID3] 预测结果: {'好瓜' if pred == 1 else '坏瓜'}")
    #
    # # 6. 全集预测准确率比较
    # custom_preds = model.predict(X_list)
    # sklearn_preds = clf.predict(X_encoded.values)
    #
    # print("\n== 准确率比较 ==")
    # print(f"[自定义 ID3] 准确率: {accuracy_score(y_encoded, custom_preds):.2f}")
    # print(f"[sklearn ID3] 准确率: {accuracy_score(y_encoded, sklearn_preds):.2f}")

