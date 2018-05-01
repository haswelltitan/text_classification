from time import time
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import numpy as np

start = time()
num = 0
raw = []
for buffer in open('train_seg.txt', encoding='utf-8').readlines()[:10000]:
    raw.append(buffer.strip())
    num += 1
for buffer in open('test_seg.txt', encoding='utf-8').readlines()[:10000]:
    raw.append(buffer.strip())

volcabulary_size = 1024
vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(raw))
pos = np.argsort(tfidftransformer.idf_)[-volcabulary_size:]
outfile = open('train_count.txt', 'w')
for line in range(0, num):
    temp = tfidf[line].toarray()
    for word in pos:
        outfile.write(str(temp[0][pos])+' ')
    outfile.write('\n')
outfile.close()
outfile = open('test_count.txt', 'w')
for line in range(num, len(raw)):
    temp = tfidf[line].toarray()
    for word in pos:
        outfile.write(str(temp[0][pos])+' ')
    outfile.write('\n')
outfile.close()

end = time()
print(end-start)
