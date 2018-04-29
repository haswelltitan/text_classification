from json import loads
from time import time
import jieba

start = time()
jieba.enable_parallel(4)
num = 0

outfile = open('train_seg.txt', 'w')
for buffer in open('train.json').readlines():
    line = ''
    for char in loads(buffer)['content']:
        if 19968 <= ord(char) <= 40869:
            line += char
    outfile.write(' '.join(jieba.cut(line, cut_all=False))+'\n')
    num += 1
    print(num)
outfile.close()

outfile = open('test_seg.txt', 'w')
for buffer in open('test.json').readlines():
    line = ''
    for char in loads(buffer)['content']:
        if 19968 <= ord(char) <= 40869:
            line += char
    outfile.write(' '.join(jieba.cut(line, cut_all=False))+'\n')
outfile.close()

jieba.disable_parallel()

end = time()
print(end-start)
