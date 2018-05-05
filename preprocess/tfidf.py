from time import time
from gensim import models, corpora
start = time()

raw = []
labels = []
for line in open('pos.txt', encoding='utf-8').readlines()[:25000]:
    raw.append(line.strip().split())
for line in open('neg.txt', encoding='utf-8').readlines()[:25000]:
    raw.append(line.strip().split())
dic = corpora.Dictionary(raw, prune_at=0)
dic.filter_extremes(no_below=10, keep_n=20000)
dic.compactify()
dic.save('dictionary.txt')
# dic = corpora.Dictionary.load('dictionary.txt')
corpus = [dic.doc2bow(line) for line in raw]
tfidf = models.TfidfModel(corpus)
tfidf.save('tfidf.model')
# tfidf = models.TfidfModel.load('tfidf.model')
outfile = open('pos_vec.txt', 'w')
for i in corpus[:25000]:
    vec = tfidf[i]
    for node in vec:
        outfile.write(str(node[0])+' '+str(node[1])+' ')
    outfile.write('\n')
outfile.close()
outfile = open('neg_vec.txt', 'w')
for i in corpus[25000:]:
    vec = tfidf[i]
    for node in vec:
        outfile.write(str(node[0])+' '+str(node[1])+' ')
    outfile.write('\n')
outfile.close()
raw = []
for line in open('test_seg.txt', 'r', encoding='utf8').readlines():
    raw.append(line.strip().split())
corpus = [dic.doc2bow(line) for line in raw]
outfile = open('test_vec.txt', 'w')
for i in corpus:
    vec = tfidf[i]
    for node in vec:
        outfile.write(str(node[0])+' '+str(node[1])+' ')
    outfile.write('\n')
outfile.close()
end = time()
print(end-start)
