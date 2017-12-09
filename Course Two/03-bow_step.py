from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
import numpy as np

#step1: load data
categories = ['alt.atheism','soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True,
                                  random_state=42)
print(twenty_train.target_names)

#step2: count  Document-term matrix
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

#print ("X_train_counts.toarray()",X_train_counts.toarray())
#count = count_vect.vocabulary_.get(u'algorithm')
#print("algorithm的出现次数为{0}".format(count))
print ("X_train_counts.shape",X_train_counts.shape)
print ("#"*50,X_train_counts.toarray()[0])
print ("#"*50)

#step3: train model
clf = SGDClassifier(loss='hinge',penalty='l2',
                    alpha=1e-3,random_state=42,
                    max_iter=5,tol=None).fit(X_train_counts,twenty_train.target) 
print ("clf",clf)
train_predicted = clf.predict(X_train_counts)
print("train_acc",np.mean(train_predicted == twenty_train.target))

#step4: test model
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)

predicted = clf.predict(X_new_counts)

for doc, category in zip(docs_new, predicted):
    print('category:%s %r => %s' % (category,doc, twenty_train.target_names[category]))


