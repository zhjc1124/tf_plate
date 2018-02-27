import numpy as np
import tensorflow as tf
import os
import cv2
from genplate import gen_plate, chars

text_, img_ = gen_plate()

img_h, img_w = 120, 32
text_len, chars_len = len(text_), len(chars)

model_path = "./model/plate/"
model_name = 'plate.ckpt'


def text2vec(text):
    vector = np.zeros(text_len*chars_len)
    for i, v in enumerate(text):
        idx = i * chars_len + chars.index(v)
        vector[idx] = 1
    return vector


def vec2text(vector):
    idxs = vector.nonzero()[0]
    text = []
    for i, v in enumerate(idxs):
        idx = v - len(chars) * i
        text.append(chars[idx])
    return ''.join(text)


def img2vec(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape != (32, 120):
        image = cv2.resize(image, (120, 32))
    return image.flatten() / 255


def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, img_h * img_w])
    batch_y = np.zeros([batch_size, text_len * chars_len])

    for i in range(batch_size):
        text, image = gen_plate()
        batch_x[i, :] = img2vec(image)
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


def init_cnn(w_alpha=0.01, b_alpha=0.1):

    def weight_variable(shape):
        initial = tf.Variable(w_alpha*tf.random_normal(shape))
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.Variable(b_alpha*tf.random_normal(shape))
        return tf.Variable(initial)

    def conv_and_pool(x_image, w, b):
        # 卷积
        h_conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_image, w, strides=[1, 1, 1, 1], padding='SAME'), b))
        # 池化
        h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # dropout减少过拟合
        return tf.nn.dropout(h_pool, keep_prob)

    x_ = tf.reshape(x, shape=[-1, img_w, img_h, 1])

    # 3层卷积层
    w_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    conv1 = conv_and_pool(x_, w_conv1, b_conv1)

    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    conv2 = conv_and_pool(conv1, w_conv2, b_conv2)

    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    conv3 = conv_and_pool(conv2, w_conv3, b_conv3)

    # 全连接层
    w_d = weight_variable([4*15*64, 1024])
    b_d = bias_variable([1024])
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = weight_variable([1024, text_len*chars_len])
    b_out = bias_variable([text_len*chars_len])
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out


def train():
    output = init_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, text_len, chars_len])

    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y, [-1, text_len, chars_len]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        step = 0
        # 读取模型
        pre = 0
        model = tf.train.latest_checkpoint(model_path)
        if model:
            saver.restore(sess, model)
            pre = int(model.split('-')[1])
            step = pre + 1
        else:
            sess.run(tf.global_variables_initializer())
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于指定值,保存模型,完成训练
                # if acc > 0.9:
                # 每10000次保存一下，避免白跑了，用bash命令 while :; do ;done
                if step >= pre + 10000:
                    saver.save(sess, model_path + model_name, global_step=step)
                    break
            step += 1


x = tf.placeholder(tf.float32, [None, img_h * img_w])
y = tf.placeholder(tf.float32, [None, text_len * chars_len])
keep_prob = tf.placeholder(tf.float32)

if __name__ == '__main__':
    train()

