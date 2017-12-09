from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

texts=["dog cat fish","dog cat cat","fish bird", 'bird']
cv = CountVectorizer()
cv_fit=cv.fit_transform(texts)

print(cv.get_feature_names())
print (cv_fit)
print(cv_fit.toarray())
#['bird', 'cat', 'dog', 'fish']
#[[0 1 1 1]
# [0 2 1 0]
# [1 0 0 1]
# [1 0 0 0]]

print(cv_fit.toarray().sum(axis=0))
#[2 3 2 2]

tfv = TfidfVectorizer()
tfv_fit=tfv.fit_transform(texts)

print("tfv.get_feature_names())",tfv.get_feature_names())
print ("tfv_fit",tfv_fit)
print("tfv_fit.toarray())",tfv_fit.toarray())

