from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import roc_auc_score
from time import time
from json import loads
from random import sample
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

k=10
train_data = neg + pos
train_label = [0] * len(neg) + [1] * len(pos)
model = rf(n_estimators=200, max_depth=50, random_state=10, min_samples_leaf=5, min_samples_split=10, n_jobs=4, max_features='sqrt')
for fold in range(1):
    pos_test = pos[(len(pos)//k)*fold:(len(pos)//k)*(fold+1)]
    # pos_train = pos[:(len(pos)//k)*fold]+pos[(len(pos)//k)*(fold+1):]
    neg_test = neg[(len(neg)//k)*fold:(len(neg)//k)*(fold+1)]
    # neg_train = neg[:(len(neg)//k)*fold]+neg[(len(neg)//k)*(fold+1):]
    # x = GridSearchCV(estimator=gbdt(learning_rate=0.1, min_samples_split=300, min_samples_leaf=20,
    #                             max_depth=150, max_features='sqrt', subsample=0.8, random_state=10),
    #              param_grid={'n_estimators': (20,101,10)}, scoring='roc_auc', iid=False)
    # model = gbdt(n_estimators=20, max_features="auto", max_depth=20, min_samples_leaf=2, min_samples_split=5, random_state=10)
    # x.fit(train_data, train_label)
    # print(x.cv_results_, x.best_params_, x.best_score_)
    model.fit(train_data, train_label)
    test_data = neg_test + pos_test
    test_label = [0] * len(neg_test) + [1] * len(pos_test)
    result = [x[1] for x in model.predict_proba(test_data)]
    print(roc_auc_score(test_label, result))
# test = open('test_vec.txt     ', 'r')
test = []
for line in open('test_vec.txt', 'r'):
    vec = [0.0] * 20000
    data = line.strip().split()
    for i in range(len(data)//2):
        vec[int(data[2*i])] = float(data[2*i+1])
    test.append(vec)
test_id = []
for line in open('test.json'):
    test_id.append(loads(line)['id'])
outfile = open('result.csv', 'w')
outfile.write('id,pred\n')
for sample in range(len(test_id)):
    outfile.write(test_id[sample]+','+str(model.predict_proba(test[sample])[1])+'\n')
outfile.close()

end = time()
print(end-start)
print('\n')
