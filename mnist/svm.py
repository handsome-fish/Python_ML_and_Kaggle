from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 从通过数据加载器获得手写体数字的数码图像数据并存储在digits变量中
# Scikit-learn中集成的手写体数字图像仅仅是完全数据中的测试数据集
digits = load_digits()

# 检视数据规模和特征维度
print(digits.data.shape)

# 随机选取75%的数据作为训练样本；其余25%的数据作为测试样本
X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.25,
                                                    random_state=33)

# 分别检视训练与测试数据规模
print(y_train.shape, y_test.shape)

# 使用数据标准化模块和支持向量机分类器LinearSVC
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC(max_iter=10000)

# 进行模型训练
lsvc.fit(X_train, y_train)
# 预测
y_predict = lsvc.predict(X_test)

# 评估
# 使用模型自带的评估函数进行准确性测评
print('The Accuracy of Linear SVC is', lsvc.score(X_test, y_test))

# 使用classification_report模块进行分析
print(classification_report(y_test,y_predict, target_names=digits.target_names.astype(str)))



