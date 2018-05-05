from time import time
from gensim.models import word2vec

start = time()
raw = []
# model = word2vec.Word2Vec.load('model.txt')
for line in open('train_seg.txt', encoding='utf-8').readlines():
    raw.append(line.strip().split())
for line in open('test_seg.txt', encoding='utf-8').readlines():
    raw.append(line.strip().split())
model = word2vec.Word2Vec(raw, size=128, window=5, min_count=10, workers=-1)
model.train(raw, epochs=10)
model.save('w2c.model')

end = time()
print(end-start)
