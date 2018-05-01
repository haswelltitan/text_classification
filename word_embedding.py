from time import time
from gensim.models import word2vec

start = time()
raw = []
num = 0
size = 32
for line in open('train_seg.txt', encoding='utf-8').readlines():
    raw.append(line.strip().split())
    num += 1
for line in open('test_seg.txt', encoding='utf-8').readlines():
    raw.append(line.strip().split())
model = word2vec.Word2Vec(raw, size=size, min_count=10, workers=-1)
model.save('model.txt')
# model = word2vec.Word2Vec.load('model.txt')

end = time()
print(end-start)
