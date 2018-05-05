from sklearn.metrics import roc_auc_score
from sklearn import svm
from time import time
from random import sample
import numpy as np
start = time()

train = open('train_vec.txt')
pos = []
neg = []
for buffer in open('train.csv', 'r').readlines()[1:]:
    temp = []
    for num in train.readline().split():
        temp.append(float(num))
    if buffer[-2] == '0':
        neg.append(temp)
    else:
        pos.append(temp)

k = 10
for fold in range(k):
    pos_test = pos[(len(pos)//k)*fold:(len(pos)//k)*(fold+1)]
    pos_train = pos[:(len(pos)//k)*fold]+pos[(len(pos)//k)*(fold+1):]
    neg_test = neg[(len(neg)//k)*fold:(len(neg)//k)*(fold+1)]
    neg_train = neg[:(len(neg)//k)*fold]+neg[(len(neg)//k)*(fold+1):]
    model = svm.SVC(probability=True)
    for epoch in range(5000):
        train_data = sample(neg_train, 64) + sample(pos_train, 64)
        train_label = [0.0] * 64 + [1.0] * 64
        model.fit(train_data, train_label)
        if epoch % 100 == 0:
            test_data = neg_test + pos_test
            test_label = [0.0] * len(neg_test) + [1.0] * len(pos_test)
            result = [x[1] for x in model.predict_proba(test_data)]
            print(epoch, roc_auc_score(test_label, result))

test = open('test_vec.txt', 'r')
test.close()

end = time()
print(end-start)
print('\n')
