from time import time
from gensim import models

start = time()
raw = []
num = 0
size = 64
for line in open('train_seg.txt', encoding='utf-8').readlines()[:100]:
    raw.append(line.strip().split())
    num += 1
for line in open('test_seg.txt', encoding='utf-8').readlines()[:100]:
    raw.append(line.strip().split())
tfidf = models.TfidfModel(raw)
tfidf.save('model.tfidf')

end = time()
print(end-start)
