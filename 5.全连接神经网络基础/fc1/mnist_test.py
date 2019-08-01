#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SECS = 5  #每检测一次后睡眠5s

def test(mnist):
    #创建一个计算图
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        #在保存模型时,若模型中采用滑动平均,则参数的滑动平均值会保存在相应文件中
        #实例化 saver 对象,实现参数滑动平均值的加载，下面三句用于加载模型的参数滑动平均(实际是加载训练时参数的滑动平均值作为本次测试的参数)
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)

        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        #argmax取出y张量的最大值，1表示按行，0表示按列
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #cast作用是数据类型转换
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            #会话被加载时模型中的所有参数被赋值为各自的滑动平均值
            with tf.Session() as sess:
                #若 ckpt 和保存的模型在指定路径中存在,则将保存的神经网络模型加载到当前会话中
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    #二个参数为 Ture 时,表示以独热码形式存取数据集
    #注意：若第一个参数指定的路径没有数据集，则会自动下载
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()
