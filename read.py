from json import loads
from time import time
import jieba.analyse
start = time()

min_sup = 5
topK = 10
train = []
dic = {}
jieba.enable_parallel(4)
for line in open('train.json').readlines()[:1000]:
    text = ""
    for char in loads(line)['content']:
        if 19968 <= ord(char) <= 40869:
            text += char
    key_words = list(jieba.analyse.extract_tags(text, topK))
    train.append(key_words)
    for word in key_words:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1
jieba.disable_parallel()


train_label = []
for buffer in open('train.csv').readlines()[1:]:
    train_label.append(int(buffer[-2]))

end = time()
print(end - start, 's')
print('\n')
