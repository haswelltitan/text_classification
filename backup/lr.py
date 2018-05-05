from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from time import time
from random import sample
import numpy as np
from json import loads
start = time()

pos = []
for line in open('pos_vec.txt', 'r'):
    vec = [0.0] * 20000
    data = line.strip().split()
    for i in range(len(data)//2):
        vec[int(data[2*i])] = float(data[2*i+1])
    pos.append(vec)
neg = []
for line in open('neg_vec.txt', 'r'):
    vec = [0.0] * 20000
    data = line.strip().split()
    for i in range(len(data)//2):
        vec[int(data[2*i])] = float(data[2*i+1])
    neg.append(vec)
model = LogisticRegression()
for epoch in range(500):
    train_data = sample(pos[:-500], 50) + sample(neg[:-500], 50)
    train_label = [1] * 50 + [0] * 50
    model.fit(train_data, train_label)
test_data = pos[-500:] + neg[-500:]
test_label = [1] * 500 + [0] * 500
result = [x[1] for x in model.predict_proba(np.array(test_data))]
print(roc_auc_score(test_label, result))

test = []
for line in open('neg_vec.txt', 'r'):
    vec = [0.0] * 20000
    data = line.strip().split()
    for i in range(len(data)//2):
        vec[int(data[2*i])] = float(data[2*i+1])
    test.append(vec)

test_id = []
for line in open('test.json').readlines():
    test_id.append(loads(line)['id'])
test = []
outfile = open('pred.csv', 'w')
outfile.write('id,pred\n')
for sample in range(len(test_id)):
    outfile.write(test_id[sample]+','+str(model.predict_proba(test[sample])[1])+'\n')
outfile.close()

end = time()
print(end-start)
print('\n')
