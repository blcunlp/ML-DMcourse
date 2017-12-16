# -*- coding:utf-8 -*-
import sys
import numpy as np
from gensim import corpora, models, similarities
import time
import matplotlib
import matplotlib.pyplot as plt
from pylab import *


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
print('现在开始训练LDA模型：---')
num_topics = 9
t_start = time.time()
# create a transformation, from tf-idf model to lda model
lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
      alpha=0.01, eta=0.01, minimum_probability=0.001, update_every = 1, chunksize = 100, passes = 1)
print('LDA模型完成，耗时 %.3f 秒' % (time.time() - t_start))

# 打印前9个文档的主题
num_show_topic = 9  # 每个文档显示前几个主题
print('下面，显示前9个文档的主题分布：')
doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布

for i in range(9):
    topic = np.array(doc_topics[i])#所有的主题中我们只选前9个
    topic_distribute = np.array(topic[:, 1])#每个文档主题分布的概率
    topic_idx = list(topic_distribute)
    print('第%d个文档的 %d 个主题分布概率分别为：' % (i, num_show_topic))
    print(topic_idx)#9个主题分布的概率

print('\n下面，显示每个主题的词分布：')
num_show_term = 7   # 每个主题下显示几个词
for topic_id in range(num_topics):
    print('第%d个主题的词与概率如下：\t' % topic_id)
    term_distribute = lda.get_topic_terms(topicid=topic_id,topn=num_show_term)
    term_distribute = np.array(term_distribute)
    term_id = term_distribute[:, 0].astype(np.int)
    print('词：\t', end='  ')
    for t in term_id:
        print(dictionary.id2token[t], end=' ')
    print('\n概率：\t', term_distribute[:, 1])

