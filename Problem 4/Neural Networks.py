
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb

num_class = 10
img_size = 28 * 28
num_neu = 512
epoch = 1000

def read(path):
    data = []
    with open(path,'r') as f:
        f.seek(0)
        for line in f:
            tmp = [float(i) for i in line.strip().split(',')]
            data.append(tmp)
    return data

def one_hot(v):
    label = []
    for i in range(len(v)):
        tmp = np.zeros(num_class)
        tmp[v[i]] = 1
        label.append(tmp)
    return label


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.1))

def model(X,w_h,w_o,b_h,b_o):
    h = tf.nn.relu(tf.add(tf.matmul(X,w_h),b_h))
    return tf.add(tf.matmul(h,w_o),b_o)

X = tf.placeholder(tf.float32,[None,img_size])
Y = tf.placeholder(tf.float32,[None,num_class])
# y_true_cls = tf.placeholder(tf.int64,[None,1])

w_h = init_weights([img_size,num_neu])
w_o = init_weights([num_neu,num_class])

b_h = init_weights([num_neu])
b_o = init_weights([num_class])

py_x = model(X, w_h, w_o, b_h, b_o)

y_pred = tf.nn.softmax(py_x)
y_pred_cls = tf.argmax(py_x,dimension=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=py_x,labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

correct_prediction = tf.equal(y_pred_cls,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_x = read("dataSets/mnist_small_train_in.txt")
train_tmp = read("dataSets/mnist_small_train_out.txt")
train_y_tmp = [int(i[0]) for i in train_tmp]

test_x = read("dataSets/mnist_small_test_in.txt")
test_tmp = read("dataSets/mnist_small_test_out.txt")
test_y_tmp = [int(i[0]) for i in test_tmp]


train_y = one_hot(train_y_tmp)
test_y = one_hot(test_y_tmp)

# print(len(train_x))
Losses = []
ACC = []
with tf.Session() as sess:

    tf.global_variables_initializer().run()
    for i in range(epoch):
        for start,end in zip(range(0,len(train_x),128),
                             range(128,len(train_x)+1,128)):
            feed_dict_train = {X:train_x[start:end],
                               Y:train_y[start:end]}
            sess.run(train_op,feed_dict=feed_dict_train)

        # loss = sess.run(cost, feed_dict=feed_dict_train)
        # print("Step:{0}, Training loss:{1}".format(i,loss))
        # Losses.append(loss)

        # if i % 20 == 0:
            # acc_train = sess.run(accuracy,feed_dict=feed_dict_train)
            # print("Step:{0}, Training accuracy:{1}".format(i,acc_train))

        feed_dict_test = {X: test_x, Y: test_y}
        acc_test = sess.run(accuracy, feed_dict_test)
        ACC.append((1-acc_test)*100)
    # print("Misclassification error on test set: {0:.3f}%".format(100*(1-acc_test)))

x_step = [i for i in range(epoch)]

plt.plot(x_step,ACC,label='Misclassification error in %')
plt.legend()
plt.show()

