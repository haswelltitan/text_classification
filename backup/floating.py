import numpy as np
infile = open('predictions-89.csv', 'r')
outfile = open('predictions.csv', 'w')
line = infile.readline()
outfile.write(line)
count = 0
for line in infile.readlines():
    id, pred = line.strip().split(',')
    result = float(pred)
    bias = np.random.rand() / 100
    if np.random.random() > 0.5:
        flag = 1
    else:
        flag = -1
    if result > 0.99:
        flag = -1
        result = 0.99
    if result < 0.01:
        flag = 1
        result = 0.01
    while bias*flag+result < 0.001 or bias*flag+result > 0.999:
        bias = np.random.rand() / 100
    outfile.write(id+','+str(bias*flag+result)+'\n')
outfile.close()
