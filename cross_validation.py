from time import time
import tensorflow as tf
start = time()

sess = tf.InteractiveSession()

# load train data
positive = []
negative = []
train_label = []
count = 0
for buffer in open('train.csv').readlines()[1:]:
    flag = int(buffer[-2])
    train_label.append(flag)
    if flag == 1:
        positive.append(count)
    else:
        negative.append(count)
    count += 1
train = []
for vector in open('train.txt').readlines()[:100000]:
    temp = []
    for element in vector.split():
        temp.append(int(element))
    train.append(temp)
end = time()
print(end-start, 's')
print('\n')
