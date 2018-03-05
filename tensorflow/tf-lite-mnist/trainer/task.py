# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import math

import tensorflow as tf
from tensorflow.python.framework import graph_util, tensor_util
from tensorflow.contrib import lite
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)


def main(args):
    """Adapted from https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_3.0_convolutional.py
    """
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

    global_step = tf.train.get_or_create_global_step()

    X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
    Y_ = tf.placeholder(tf.float32, [None, 10])
 
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * tf.exp(-tf.to_float(global_step) / decay_speed)

    K = 4  # first convolutional layer output depth
    L = 8  # second convolutional layer output depth
    M = 12  # third convolutional layer
    N = 200  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.ones([K])/10)
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.ones([L])/10)
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.ones([M])/10)

    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.ones([N])/10)
    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.ones([10])/10)

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits, name='output')

    # ops from this point and on will not be frozen nor converted to tflite
    # loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100

    # *training* accuracy
    correct_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), dtype=tf.float32))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

    # saver
    # keep all checkpoints
    saver = tf.train.Saver(max_to_keep=args.n_steps/args.save_model_steps)

    # init
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in xrange(1, args.n_steps+1):
            batch_X, batch_Y = mnist.train.next_batch(args.batch_size)
            _, ccount, gstep = sess.run([train_step, correct_count, global_step], {X: batch_X, Y_: batch_Y})

            if i % 10 == 0:
                print('batch {}, train accuracy: {}'.format(i, ccount / args.batch_size))

            if i % args.save_model_steps == 0:
                # save checkpoints
                print('saving checkpoint')
                checkpoint_path = os.path.join(args.output_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=gstep)

                # save frozen graph_def
                print('saving frozen graph')
                output_node_names = [Y.op.name]
                frozen_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
                tf.train.write_graph(frozen_graph_def, args.output_dir, 'model.pb', as_text=False)

                # save tf-lite model
                print('saving tf-lite model')
                tflite_model_path = os.path.join(args.output_dir, 'model.tflite')

                # switch to a temporary graph to change tensor shapes
                with tf.Graph().as_default():
                    for node in frozen_graph_def.node:
                        if 'shape' in node.attr:
                            node.attr['shape'].shape.dim[0].size = 1

                    # If we leave the default name=None, the imported tensor names get the `import/` prefix, which interferes with toco_convert.
                    _X, _Y = tf.import_graph_def(frozen_graph_def, return_elements=['input:0', 'output:0'], name='')

                    tflite_model = lite.toco_convert(frozen_graph_def, input_tensors=[_X], output_tensors=[_Y])
                    with tf.gfile.FastGFile(tflite_model_path, 'wb') as f:
                        f.write(tflite_model)

        # test data
        test_X = mnist.test.images
        test_Y = mnist.test.labels
        test_ccount = sess.run(correct_count, {X: test_X, Y_: test_Y})
        print('>>> test accuracy: {}'.format(test_ccount / len(test_Y)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n-steps',
        type=int,
        default=3000)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100)
    parser.add_argument(
        '--save-model-steps',
        type=int,
        default=100)
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/tflite_output')
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/tflite_output')

    args = parser.parse_args()

    main(args)