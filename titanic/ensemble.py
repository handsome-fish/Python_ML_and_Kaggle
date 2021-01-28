# 2.1.1.6 集成模型
# 对比单一决策树与集成模型中随机森林分类器以及梯度提升决策树的性能差异

# 导入pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


titanic = pd.read_csv("http://biostat.mc.vanderdilt.edu/wiki/pub/Main/DataSets/titanic.txt")
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

# 补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(), inplace=True)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# 特征抽取
vec = DictVectorizer(sparse=False)

# 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 使用单一决策树进行模型训练以及预测分析
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)

# 使用随机森林分类器进行集成模型的训练以及预测分析
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 使用梯度提升决策树进行集成模型的训练以及预测分析
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)

# 输出单一决策树在测试集上的分类准确性，以及更加详细的准确率、召回率、F1指标
print("The accuracy of decision tree is", dtc.score(X_test, y_test))
print(classification_report(y_test, dtc_y_pred, target_names=["died", "survived"]))

# 输出随机森林分类器结果
print("The accuracy of random forest classifier is", rfc.score(X_test, y_test))
print(classification_report(y_test, rfc_y_pred, target_names=["died", "survived"]))

# 输出梯度提升决策树结果
print("The accuracy of gradient boosting is", gbc.score(X_test, y_test))
print(classification_report(y_test, gbc_y_pred, target_names=["died", "survived"]))
