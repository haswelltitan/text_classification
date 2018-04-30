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
top = sorted(dic.keys(), key=dic.__getitem__, reverse=True)[:dic_size-1]
dic = {}
for word in top:
    dic[word] = len(dic) + 1
clean = []
for line in raw:
    temp = []
    for word in line:
        if word in dic:
            temp.append(dic[word])
        else:
            temp.append(0)
    clean.append(temp)
del raw

count = [['UNK', -1]]
for word in range(len(top)):
    count.append([top[word], dic[top[word]]])
print(len(count))
for line in raw:
    data = []
end = time()
print(end-start)
