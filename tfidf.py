from time import time
from sklearn.feature_extraction.text import TfidfTransformer as tfidf, CountVectorizer as counter

start = time()
num = 0
raw = []
dic = {}
for buffer in open('train_seg.txt', encoding='utf-8').readlines():
    raw.append(buffer.strip().split(' '))
    num += 1
for buffer in open('test_seg.txt', encoding='utf-8').readlines():
    raw.append(buffer.strip().split(' '))

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

outfile = open('train_vec.txt', 'w')
for count in range(0, num):
    c = counts[count]
    for word in pos:
        outfile.write(str(round(c[word], 3))+' ')
    outfile.write('\n')
outfile.close()
outfile = open('test_vec.txt', 'w')
for count in range(num, len(counts)):
    c = counts[count]
    for word in pos:
        outfile.write(str(round(c[word], 3)) + ' ')
    outfile.write('\n')
outfile.close()

end = time()
print(end-start)
