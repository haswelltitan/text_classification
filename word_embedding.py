from time import time

start = time()
num = 0
raw = []
dic = {}
dic_size = 128
for buffer in open('train_seg.txt', encoding='utf-8').readlines()[:100]:
    raw.append(buffer.strip().split())
    num += 1
for buffer in open('test_seg.txt', encoding='utf-8').readlines()[:100]:
    raw.append(buffer.strip().split())
for line in raw:
    for word in line:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1
top_words = sorted(dic.keys(), key=dic.__getitem__, reverse=True)[:dic_size-1]
count = [['UNK', -1]]
for word in top_words:
    count.append([word, dic[word]])
print(len(count))

end = time()
print(end-start)
