#!/usr/bin/env python

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


import os
import pickle
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util, graph_util
from tensorflow.examples.tutorials.mnist import input_data

from helpers import load_pb_as_graph_def


MODEL_DIR = 'model'
MODEL_FN = 'mobilenet_v1_1.0_224/frozen_graph.pb'

MODEL_INFO = {
    'input_name': 'input:0',
    # shape=(1, 1, 1, 1024):
    'bottleneck_name': 'MobilenetV1/Logits/AvgPool_1a/AvgPool:0',
    'bottleneck_op': 'MobilenetV1/Logits/AvgPool_1a/AvgPool'
}

def main(args):
    tf.reset_default_graph()

    mnist_input = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
    rgb = tf.image.grayscale_to_rgb(mnist_input)
    resized_input = tf.image.resize_bilinear(rgb, (224, 224))

    graph_def = load_pb_as_graph_def(os.path.join(MODEL_DIR, MODEL_FN))

    # Get only the subgraph up to the bottleneck op.
    graph_def = graph_util.extract_sub_graph(graph_def, [MODEL_INFO['bottleneck_op']])

    # Modify the tensors in the graph to allow unknown batch size.
    # (The frozen graph has batch size set to 1.)
    for node in graph_def.node:
        if 'shape' in node.attr:
            node.attr['shape'].shape.dim[0].size = -1

    input_,  bottleneck = tf.import_graph_def(
        graph_def, name='',
        return_elements=[MODEL_INFO['input_name'], MODEL_INFO['bottleneck_name']])

    bottleneck = tf.reshape(bottleneck, (-1, 1024))

    w = tf.get_variable(name='last_weight', dtype=tf.float32, shape=(1024, 10))
    b = tf.get_variable(name='last_bias', dtype=tf.float32, shape=(10))

    logits = tf.matmul(bottleneck, w) + b

    labels = tf.placeholder(dtype=tf.float32, shape=(None, 10))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    # for evaluation
    indices = tf.argmax(logits, axis=1)
    one_hots = tf.one_hot(indices, depth=10)
    correct = tf.reduce_sum(one_hots * labels)

    # for training
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    mnist = input_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(args.n_iter):
            imgs, lbls = mnist.train.next_batch(args.batch_size)

            _resized_input = sess.run(resized_input, {mnist_input: imgs})

            feed_dict = {
                input_: _resized_input,
                labels: lbls
            }
            _loss, _ = sess.run([loss, train_op], feed_dict)

            print('batch {}, mean loss {}'.format(i, _loss))
                

        print('test:')

        imgs = mnist.test.images
        lbls = mnist.test.labels

        _resized_input = sess.run(resized_input, {mnist_input: imgs})

        feed_dict = {
            input_: _resized_input,
            labels: lbls
        }

        _correct = sess.run(correct, feed_dict)

        pass

        # test_size = 1000
        # test_batch_count = 0

        # total_test_size = 0
        # total_correct = 0
        # while mnist.test.epochs_completed == 0:
        #     imgs, lbls = mnist.test.next_batch(test_size)

        #     _resized_input = sess.run(resized_input, {mnist_input: imgs})

        #     feed_dict = {
        #         input_: _resized_input,
        #         labels: lbls
        #     }

        #     _correct = sess.run(correct, feed_dict)

        #     print('batch {}: {} / {}'.format(test_batch_count, _correct, test_size))

        #     test_batch_count += 1
        #     total_test_size += test_size
        #     total_correct += _correct

        # print('total: {} / {}'.format(total_correct, total_test_size))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--n-iter', type=int, default=3)

    args = parser.parse_args()

    main(args)

