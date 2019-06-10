# LanguageDetectorWithNaiveBayes
- 语种检测可以看做一个多文本分类问题.文本分类是nlp中最普遍的一种任务,通常包含三个部分:
  1. 数据预处理(包括数据清洗,分词等)
  2. 文本表示(常用的方法有词袋,TF-IDF,N-gram,word-vec,word embedding等)
  3. 建立模型(包括机器学习和深度学习方法,如LR,SVM,NB,textCNN,textRNN,LSTM等)
- 这里我们使用朴素贝叶斯(`Naive Bayes`, `NB`)来实现多语种分类问题.
## 数据预处理
  1. 不同类别的样本数尽量相同或相近
  2. 使用正则表达式(regular expression)移除特殊符号,如`@`,`#`, 网址等
  3. 由于要检测的语种都是拉丁语系的语言,需要从字母粒度进行分析,所以按照字母进行划分
## 文本表示
由于我们在字母粒度进行向量表示,这里分别使用TF-IDF和N-gram进行文本表示.相比于词袋模型只考虑了word是否出现,TF-IDF对不同的词计算不同的权重,而N-gram考虑了文本中词出现的先后顺序关系.
  1. N-gram使用sklearn.feature_extraction.text.CountVectorizer()
  2. TF-IDF使用sklearn.feature_extraction.text.TfidfVectorizer()
## 朴素贝叶斯
- 朴素贝叶斯方法是基于贝叶斯理论和条件独立假设的监督学习算法的集合.我们使用`sklearn.naive_bayes.MultinomialNB()`来实现多项式朴素贝叶斯模型.详细参阅[sklearn中朴素贝叶斯的使用说明](https://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes)

