from json import loads
from jieba import cut
from time import time
start = time()

train = []
dic = {}
for line in open('train.json').readlines()[:10000]:
    text = ""
    for char in loads(line)['content']:
        if 19968 <= ord(char) <= 40869:
            text += char
    train.append(text)
    segs = cut(text, True)
    for word in segs:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1

train_label = []
for buffer in open('train.csv').readlines()[1:]:
    train_label.append(int(buffer[-2]))

end = time()
print(end - start, 's')
