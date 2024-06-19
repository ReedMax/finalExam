from flask import Flask, render_template, request, send_file, make_response
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
from collections import Counter

app = Flask(__name__)

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# 数据归一化函数
def normalize(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        distances = self._compute_distances(X_test)
        return self._predict_labels(distances)

    def _compute_distances(self, X_test):
        # 使用广播和向量化计算距离
        distances = np.sqrt(np.sum((X_test[:, np.newaxis] - self.X_train) ** 2, axis=2))
        return distances

    def _predict_labels(self, distances):
        # 找到每个测试样本的最近k个训练样本
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_indices]
        # 返回出现次数最多的标签
        most_common = [Counter(labels).most_common(1)[0][0] for labels in k_nearest_labels]
        return np.array(most_common)



# 加载数据集
wine = load_wine()
data = wine.data
target = wine.target
feature_names = wine.feature_names


# 合并数据和标签以传递给模板
wine_data = np.column_stack((data, target))

# 用于存储图像数据的全局变量
figures = {}

@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names, wine_data=wine_data, enumerate=enumerate)

@app.route('/train', methods=['POST'])
def train():
    global figures

    # 获取用户选择的特征
    feature1 = int(request.form['feature1'])
    feature2 = int(request.form['feature2'])

    # 确保两个特征不同
    if feature1 == feature2:
        return "选择的特征不能相同，请返回重选", 400

    selected_features = [feature1, feature2]

    # 数据归一化
    normalized_data = normalize(data)

    # 选择用户选择的两个特征
    data_2d = normalized_data[:, selected_features]

    # 划分训练集和验证集，70%用于训练，30%用于测试
    X_train, X_test, y_train, y_test = train_test_split(data_2d, target, test_size=0.3, random_state=42)

    models = {
        'K-近邻,k值为3': CustomKNN(k=3),
        'K-近邻,k值为5': CustomKNN(k=5),
        '逻辑回归': LogisticRegression(max_iter=200),
        '决策树': DecisionTreeClassifier(),
        '线性判别分析': LinearDiscriminantAnalysis(),
        '朴素贝叶斯': GaussianNB(),
        '支持向量机': SVC(kernel='linear', probability=True),
        '随机森林': RandomForestClassifier()
    }

    results = {}  # 存储每个模型的准确率
    figures = {}  # 存储每个模型的决策边界图像

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy

        # 绘制决策边界
        fig, ax = plt.subplots(figsize=(10, 8))  # 调整图表尺寸
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))


        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3)

        # 绘制训练数据和测试数据
        sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, ax=ax, palette="deep", s=50)  # 调整点的大小
        sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=predictions, ax=ax, marker='X', s=100, palette="deep")
        ax.set_title(f'{name} 决策边界 (准确率: {accuracy:.2f})')
        ax.set_xlabel(feature_names[feature1])  # 设置横轴标签
        ax.set_ylabel(feature_names[feature2])  # 设置纵轴标签
        figfile = io.BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figures[name] = figfile
        plt.close(fig)

    response = make_response(render_template('results.html', results=results, figures=figures.keys()))
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response

@app.route('/fig/<name>')
def fig(name):
    figfile = figures.get(name)
    if figfile is None:
        return "Figure not found", 404
    figfile.seek(0)
    return send_file(figfile, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
