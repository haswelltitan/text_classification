from json import loads
from time import time
import jieba

start = time()
stop_words = []
for line in open('stop_words.txt', encoding='utf_8').readlines():
    stop_words.append(line.strip())

outfile = open('train_seg.txt', 'w')
for buffer in open('train.json', encoding='utf-8').readlines():
    line = ''
    for char in loads(buffer)['content']:
        if 19968 <= ord(char) <= 40869:
            line += char
    raw = list(jieba.cut(line, cut_all=False))
    for word in stop_words:
        if word in raw:
            raw.remove(word)
    outfile.write(' '.join(raw)+'\n')
outfile.close()

outfile = open('test_seg.txt', 'w')
for buffer in open('test.json', encoding='utf-8').readlines():
    line = ''
    for char in loads(buffer)['content']:
        if 19968 <= ord(char) <= 40869:
            line += char
    raw = list(jieba.cut(line, cut_all=False))
    for word in stop_words:
        if word in raw:
            raw.remove(word)
    outfile.write(' '.join(raw) + '\n')
outfile.close()

end = time()
print(end-start)
