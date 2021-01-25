# 2.1.1.1 线性分类器

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# 使用pandas.read_csv函数从互联网读取指定数据
# data = pd.read_csv(
#     'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
#     names=column_names)
data = pd.read_csv(
    '../datasets/breast-cancer-wisconsin.data',
    names=column_names)

# 将?替换为标准缺失值表示
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）
data = data.dropna(how='any')

# 设置列不限制数量
pd.set_option('display.max_columns', None)
# 输出data的数据量和维度
# print(data.head())

# 使用sklearn.cross_valiation里的train_test_split模块用于分割数据
# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                    data[column_names[10]],
                                                    test_size=0.25,
                                                    random_state=33)

# 查验训练和测试样本的数量和类别分布
# print(y_train.value_counts())
# print(y_test.value_counts())

# 使用逻辑斯蒂回归与随机梯度参数估计两种方法对上述处理后的训练数据进行学习
# 标准化数据，保证每个维度的特征数据方差为1，均值为0.使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
# 对testData使用和trainData同样的均值、方差、最大最小值等指标进行转换, 从而保证train、test处理方式相同
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化LogisticRegression与SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()

# 调用LogisticRegression中的fit函数/模块用来训练模型参数
lr.fit(X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果储存在变量lr_y_predict中
lr_y_predict=lr.predict(X_test)

# 调用SGDClassifier中的fit函数用来训练模型参数
sgdc.fit(X_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果储存在变量sgdc_y_predict中
sgdc_y_predict = sgdc.predict(X_test)

# 对两个模型进行性能分析

# 使用逻辑斯蒂回归自带的评分函数score获取模型在测试集上的准确性结果
print("Accuracy of LR Classifier:", lr.score(X_test,y_test))
# 利用classification_report模块获得LogisticRegression其他三个指标的结果
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

# 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果
print("Accuracy of SGD Classifier:", sgdc.score(X_test, y_test))
# 利用classification_report模块获得SGDClassifier其他三个指标的结果
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))

l = np.array(y_test) - lr_y_predict
j = 0
for i in l:
    if i != 0:
        print(i, lr_y_predict[j])
    j += 1
