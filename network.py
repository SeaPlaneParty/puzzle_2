import time
import pickle
import tensorflow as tf

DATA_PATH = './dataset.pkl'
RANDOM_SEAD = time.time()

tf.set_random_seed(RANDOM_SEAD)

data = pickle.load(open(DATA_PATH, 'rb'))

x = tf.placeholder(shape=(None, 32), dtype=tf.float32)
y = tf.placeholder(shape=(None), dtype=tf.int64)

#make network
h_1 = tf.layers.dense(x, units=10000, activation='relu')
y_hat = tf.layers.dense(h_1, units=2, activation='softmax')
acc = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(y_hat, axis=-1), y), tf.float32))

#run "training"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out = sess.run(y_hat, feed_dict={x: data['x']})
    print(sess.run(acc, feed_dict={x: data['x'], y: data['y']}))
