from json import loads
from time import time
import jieba.analyse
start = time()

min_sup = 10
topK = 10
train = []
dic = {}
jieba.enable_parallel(4)

# exclude numbers & irrelevant characters
for line in open('train.json').readlines():
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

test = []
for line in open('test.json').readlines():
    text = ""
    for char in loads(line)['content']:
        if 19968 <= ord(char) <= 40869:
            text += char
    key_words = list(jieba.analyse.extract_tags(text, topK))
    test.append(key_words)
    for word in key_words:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1
jieba.disable_parallel()

# find set of frequent words
top_words = []
for word in dic.keys():
    if dic[word] >= min_sup:
        top_words.append(word)

# extract features for train and test data
train_labels = []
file = open('train.txt', 'w')
for text in train:
    for word in top_words:
        file.write(str(text.count(word))+' ')
    file.write('\n')
file.close()

test_labels = []
file = open('test.txt', 'w')
for text in test:
    for word in top_words:
        file.write(str(text.count(word))+' ')
    file.write('\n')
file.close()

# train_label = []
# for buffer in open('train.csv').readlines()[1:]:
#     train_label.append(int(buffer[-2]))

end = time()
print(end - start, 's')
print('\n')
