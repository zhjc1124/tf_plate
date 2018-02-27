from ocr_train import *
import cv2

output = init_cnn()
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint(model_path))
predict = tf.argmax(tf.reshape(output, [-1, text_len, chars_len]), 2)


def ocr(plate):
    text_list = sess.run(predict, feed_dict={x: [plate], keep_prob: 1})
    text = text_list[0].tolist()
    vector = np.zeros(len(text) * len(chars))
    i = 0
    for n in text:
        vector[i * chars_len + n] = 1
        i += 1
    return vec2text(vector)


if __name__ == '__main__':
    plate_dir = './plate/'
    # for i in range(100):
    #     img = cv2.imread(plate_dir+'%02d.jpg' % i)
    #     img = img2vec(img)
    #     print(ocr(img))
    # img = cv2.imread('/home/zhc1124/音乐/test.jpg')
    # img = img2vec(img)
    # print(ocr(img))
    time = 0
    for i in range(1000):
        text, img = gen_plate()
        img = img2vec(img)
        pre = ocr(img)
        if text != pre:
            time += 1
        print(text, pre)
    print(time)
    sess.close()
