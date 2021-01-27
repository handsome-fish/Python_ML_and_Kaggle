# 2.1.1.4 K近邻

# 导入iris数据加载器
from sklearn.datasets import load_iris
# 数据分割
from sklearn.model_selection import train_test_split
# 导入标准化模块
from sklearn.preprocessing import StandardScaler
# 导入K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入评估模块
from sklearn.metrics import classification_report

# 使用数据加载器读取数据
iris = load_iris()

# 查验数据规模
print(iris.data.shape)

# 查看数据说明。对于一名机器学习的实践者来讲，这是个好习惯
print(iris.DESCR)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

# 对数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 训练并预测
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

# 评估
# 使用自带的评估函数评估
print("The accuracy of K-Nearest Neighbour Classifier is", knc.score(X_test, y_test))

# 详细分析
print(classification_report(y_test, y_predict, target_names=iris.target_names))
