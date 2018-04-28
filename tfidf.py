from json import loads
from time import time
from sklearn.feature_extraction.text import TfidfTransformer as tfidf, CountVectorizer as counter
import jieba


start = time()
num = 0
raw = []
dic = {}
jieba.enable_parallel(4)
for buffer in open('train.json').readlines():
    line = ''
    for char in loads(buffer)['content']:
        if 19968 <= ord(char) <= 40869:
            line += char
    raw.append(' '.join(jieba.cut(line, cut_all=False)))
    num += 1
for buffer in open('test.json').readlines():
    line = ''
    for char in loads(buffer)['content']:
        if 19968 <= ord(char) <= 40869:
            line += char
    raw.append(' '.join(jieba.cut(line, cut_all=False)))
jieba.disable_parallel()

clean = counter()
counts = tfidf().fit_transform(clean.fit_transform(raw)).toarray()
words = clean.get_feature_names()
for count in range(len(counts)):
    c = counts[count]
    for word in range(len(words)):
        w = words[word]
        if c[word] != 0:
            if w in dic:
                dic[w] += c[word]
            else:
                dic[w] = c[word]
top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:256]
pos = []
for i in top:
    pos.append(words.index(i[0]))

outfile = open('train.txt', 'w')
for count in range(0, num):
    c = counts[count]
    for word in pos:
        outfile.write(str(round(c[word], 3))+' ')
    outfile.write('\n')
outfile.close()
outfile = open('test.txt', 'w')
for count in range(num, len(counts)):
    c = counts[count]
    for word in pos:
        outfile.write(str(round(c[word], 3)) + ' ')
    outfile.write('\n')
outfile.close()

end = time()
print(end-start)
