from time import time
from gensim.models import word2vec

start = time()
raw = []
num = 0
size = 64
for line in open('train_seg.txt', encoding='utf-8').readlines():
    raw.append(line.strip().split())
    num += 1
for line in open('test_seg.txt', encoding='utf-8').readlines():
    raw.append(line.strip().split())
model = word2vec.Word2Vec(raw, size=size, min_count=5, workers=-1)
model.save('model.txt')
# model = word2vec.Word2Vec.load('model.txt')
# outfile = open('train_vec.txt', 'w')
# for line in raw[:num]:
#     vec = np.zeros(size)
#     for word in line:
#         if word in model:
#             vec = vec + model[word]
#     for word in vec:
#         outfile.write(str(round(word, 4))+' ')
#     outfile.write('\n')
# outfile.close()
# outfile = open('test_vec.txt', 'w')
# for line in raw[num:]:
#     vec = np.zeros(size)
#     for word in line:
#         if word in model:
#             vec = vec + model[word]
#     for word in vec:
#         outfile.write(str(round(word, 4))+' ')
#     outfile.write('\n')
# outfile.close()

end = time()
print(end-start)
