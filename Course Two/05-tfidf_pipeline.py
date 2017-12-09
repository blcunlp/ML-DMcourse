import numpy as np
from sklearn.datasets import load_files
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

#step1: read data
categories = ['alt.atheism','soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True,
                                  random_state=42)

print(twenty_train.target_names)

#step2: count
#step3: tf-idf
#step4: train classfier with svm
#step4.1: build a pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge',penalty='l2',
                                          alpha=1e-3,random_state=42,
                                          max_iter=5,tol=None)),
                    ])
#step4.2: train the model
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
print(text_clf)

#step5: evaluate on the test set
twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories,
                                 shuffle=True,
                                 random_state=42)

docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
# 计算预测结果的准确率
print("准确率为：")
print(np.mean(predicted == twenty_test.target))
