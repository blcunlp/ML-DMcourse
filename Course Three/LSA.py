# -*- coding:utf-8 -*-

import numpy as np
from gensim import corpora, models, similarities
import time



def load_stopword():
    f_stop = open('sw.txt', encoding='utf-8')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw

# remove the stop words
sw = load_stopword()
f = open('news_cn.dat', encoding='utf-8')    # load the text data
texts = [[word for word in line.strip().lower().split() if word not in sw] for line in f]
f.close()
M = len(texts)
print('语料库载入完成，据统计，一共有 %d 篇文档' % M)

# build the dictionary for texts
dictionary = corpora.Dictionary(texts)
dict_len = len(dictionary)
print("dict_len:%d"%(dict_len))
# transform the whole texts to sparse vector
corpus = [dictionary.doc2bow(text) for text in texts]
print("corpus_len:%d"%(len(corpus)))
# create a transformation, from initial model to tf-idf model
corpus_tfidf = models.TfidfModel(corpus)[corpus]
print('现在开始创建LSI模型：---')
num_topics = 9
t_start = time.time()
# create a transformation, from tf-idf model to lda model
lsi=models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=9)
print('LSI模型完成，耗时 %.3f 秒' % (time.time() - t_start))

topics=lsi.show_topics(num_words=7,log=0) 
for tpc in topics: 
  print(tpc)

topic_last=lsi[corpus_tfidf[-1]]
print(topic_last)
