from json import loads
from time import time
import jieba.analyse
start = time()

dic = {}
train = []
for line in open('train.json').readlines():
    text = ""
    for char in loads(line)['content']:
        if 19968 <= ord(char) <= 40869:
            text += char
    key_words = list(jieba.analyse.extract_tags(text))
    train.append(key_words)
    for word in key_words:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1
test = []
for line in open('test.json').readlines():
    text = ""
    for char in loads(line)['content']:
        if 19968 <= ord(char) <= 40869:
            text += char
    key_words = list(jieba.analyse.extract_tags(text))
    test.append(key_words)
    for word in key_words:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1

top_words = sorted(dic.keys(), key=dic.__getitem__, reverse=True)[:256]
file = open('train.txt', 'w')
for text in train:
    for pos in range(len(top_words)):
        if top_words[pos] in text:
            file.write(str(pos)+' ')
    file.write('\n')
file.close()
file = open('test.txt', 'w')
for text in test:
    for pos in range(len(top_words)):
        if top_words[pos] in text:
            file.write(str(pos)+' ')
    file.write('\n')
file.close()

end = time()
print(end-start, 's')
print('\n')
