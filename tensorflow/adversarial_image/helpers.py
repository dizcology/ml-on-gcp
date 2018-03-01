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

import tensorflow as tf
from tensorflow.python.framework import graph_util, tensor_util


def load_pb_as_graph_def(frozen_graph_fn):
    graph_def = tf.GraphDef()

    with open(frozen_graph_fn, 'rb') as f:
        graph_def.ParseFromString(f.read())

    return graph_def


def save_graph_as_pb(graph, logdir, name):
    tf.train.write_graph(graph, logdir, name, as_text=False)


# cf. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
def freeze_graph_def(sess, output_node_names):
    return graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)


def get_constant(constant_node_def):
    return tensor_util.MakeNdarray(constant_node_def.attr['value'].tensor)

def update_constant(node_def, new_value):
    """Updates in place!
    """
    tensor_proto = tensor_util.make_tensor_proto(new_value)
    attr_value = tf.AttrValue(tensor=tensor_proto)
    node_def.attr['value'].CopyFrom(attr_value)

    return node_def

# requires graph_def *and* input/output tensors
def make_tflite_model(graph_def, input_tensors, output_tensors):
    return tf.contrib.lite.toco_convert(graph_def, input_tensors, output_tensors)


