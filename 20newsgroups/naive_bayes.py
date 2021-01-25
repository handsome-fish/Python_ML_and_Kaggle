# 2.1.1.3 朴素贝叶斯

# 获取数据
from sklearn.datasets import fetch_20newsgroups
# 导入train_test_split
from sklearn.model_selection import train_test_split
# 导入文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
# 导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 用于详细的分类性能报告
from sklearn.metrics import classification_report


# 该数据需要即使从互联网下载数据
news = fetch_20newsgroups(subset='all')

# 检查数据规模和细节
print(len(news.data))
print(news.data[0])


# 随机25%作为测试集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# 使用朴素贝叶斯分类器对新闻文本数据进行类别预测
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 初始化NB模型
mnb = MultinomialNB()

# 利用训练数据对模型参数进行估计
mnb.fit(X_train, y_train)
# 对测试样本进行类别预测
y_predict = mnb.predict(X_test)

# 评估
print('The accuracy of Naive Bayes Classifier is', mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))
