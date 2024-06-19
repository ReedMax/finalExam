import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 数据归一化函数
def normalize(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)

# 加载数据集
wine = load_wine()
print(wine.data)
data = wine.data
target = wine.target

# 数据归一化
normalized_data = normalize(data)

# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(normalized_data, target, test_size=0.3, random_state=42)

# 初始化模型
models = {
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(),
    'LDA': LinearDiscriminantAnalysis(),
    'Naive Bayes': GaussianNB()
}

# 训练和测试模型
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy
    print(f'{name} Accuracy: {accuracy}')

# 找出最佳分类器
best_model_name = max(results, key=results.get)
print(f'Best model: {best_model_name} with accuracy {results[best_model_name]}')

# 尝试调优最佳模型（示例中调优KNN的k值）
if best_model_name == 'KNN':
    best_accuracy = 0
    best_k = 1
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    print(f'Best KNN k: {best_k} with accuracy {best_accuracy}')

# 绘制决策边界函数
def plot_decision_boundary(model, X, y, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
    plt.title(title)
    plt.show()

# 由于数据维度较高，我们选择前两个特征进行决策边界绘制
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

for name, original_model in models.items():
    model = original_model.__class__()  # 创建模型的新实例
    model.fit(X_train_2d, y_train)
    plot_decision_boundary(model, X_train_2d, y_train, f'{name} Decision Boundary')
