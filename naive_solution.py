from time import time
import tensorflow as tf
import random
from json import loads
start = time()

# load train data
positive = []
negative = []
train_label = []
count = 0
for buffer in open('train.csv').readlines()[1:]:
    flag = int(buffer[-2])
    if flag == 0:
        label = [1, 0]
    else:
        label = [0, 1]
    train_label.append(label)
    if count < 320000:
        if flag == 0:
            negative.append(count)
        else:
            positive.append(count)
    count += 1
train = []
for vector in open('train.txt').readlines():
    temp = [0] * 256
    for element in vector.strip().split():
        temp[int(element)] = 1
    train.append(temp)

# tensor graph
sess = tf.InteractiveSession()
in0 = tf.placeholder("float", shape=[None, 256])
out0 = tf.placeholder("float", shape=[None, 2])
in1 = tf.reshape(in0, [-1, 16, 16, 1])
w1 = tf.Variable(tf.truncated_normal([4, 4, 1, 10], stddev=0.1))
b1 = tf.constant(0.1, shape=[10])
f1 = tf.nn.relu(tf.nn.conv2d(in1, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
p1 = tf.nn.max_pool(f1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
w2 = tf.Variable(tf.truncated_normal([4, 4, 10, 20], stddev=0.1))
b2 = tf.constant(0.1, shape=[20])
f2 = tf.nn.relu(tf.nn.conv2d(p1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
p2 = tf.nn.max_pool(f2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
mid = tf.reshape(p2, [-1, 4*4*20])
w3 = tf.Variable(tf.truncated_normal([4*4*20, 256], stddev=0.1))
b3 = tf.constant(0.1, shape=[256])
f3 = tf.nn.relu(tf.matmul(mid, w3) + b3)
drop_prob = tf.placeholder("float")
f4 = tf.nn.dropout(f3, drop_prob)
w4 = tf.Variable(tf.truncated_normal([256, 2], stddev=0.1))
b4 = tf.constant(0.1, shape=[2])
out1 = tf.nn.softmax(tf.matmul(f4, w4) + b4)
cross_entropy = -tf.reduce_sum(out0 * tf.log(out1+1e-20))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct = tf.equal(tf.argmax(out1, 1), tf.argmax(out0, 1))
acc = tf.reduce_mean(tf.cast(correct, "float"))
_, auc = tf.metrics.auc(out0, out1)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for epoch in range(50000):
    choose = random.sample(positive, 64)+random.sample(negative, 64)
    random.shuffle(choose)
    data, label = [train[i] for i in choose], [train_label[i] for i in choose]
    train_step.run(feed_dict={in0: data, out0: label, drop_prob: 0.5})
    if epoch % 1000 == 0:
        print(sess.run(auc, feed_dict={in0: train[320000:321000], out0: train_label[320000:321000], drop_prob: 0.5}))

test_id = []
for line in open('test.json').readlines():
    test_id.append(loads(line)['id'])
test = []
for vector in open('test.txt').readlines():
    temp = [0] * 256
    for element in vector.strip().split():
        temp[int(element)] = 1
    test.append(temp)
outfile = open('result.csv', 'w')
outfile.write('id,pred\n')
pred = sess.run(fetches=out1, feed_dict={in0: test, drop_prob: 1.0})
for sample in range(len(test_id)):
    outfile.write(test_id[sample]+','+str(pred[sample][1])+'\n')
outfile.close()
end = time()
print(end-start, 's')
print('\n')
