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

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image

import os
import urllib
import tarfile
import pickle

import tensorflow as tf
from tensorflow.contrib.graph_editor import reroute

key_url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'
#urllib.urlretrieve(key_url, 'imagenet1000_clsid_to_human.pkl')

with open('imagenet1000_clsid_to_human.pkl') as kf:
	_key = pickle.load(kf)

# off by 1
key = {0: 'unknown'}
for k, v in _key.iteritems():
	key[k+1] = _key[k]


model_info = {
	'data_url': 'http://download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz',
	'input_name': 'input:0',
	'output_name': 'MobilenetV1/Predictions/Softmax:0',
	'logits_name': 'MobilenetV1/Logits/SpatialSqueeze:0'
}
model_dir = 'model'
frozen_graph_fn = os.path.join(model_dir, 'mobilenet_v1_1.0_224', 'frozen_graph.pb')

def download_and_extract(data_url, model_dir=model_dir):
	os.makedirs(model_dir)
	filename = data_url.split('/')[-1]
	filepath = os.path.join(model_dir, filename)

	urllib.urlretrieve(data_url, filepath)

	tarfile.open(filepath, 'r:gz').extractall(model_dir)


def load_pb_as_graph_def(frozen_graph_fn):
	graph_def = tf.GraphDef()

	with open(frozen_graph_fn, 'rb') as f:
		graph_def.ParseFromString(f.read())

	return graph_def



epsilon = 0.2
n_iter = 100

if __name__ == '__main__':
	original = plt.imread('fish1.jpg')
	a = resize(original, (224, 224, 3), mode='constant')

	tf.reset_default_graph()

	graph_def = load_pb_as_graph_def(frozen_graph_fn)
	input_, logits, prob = tf.import_graph_def(
		graph_def, name='',
		return_elements=[model_info['input_name'], model_info['logits_name'], model_info['output_name']])

	# create a variable and reroute from it
	var_input = tf.get_variable(name='var_input', dtype=input_.dtype, shape=input_.shape)
	reroute._reroute_t(var_input.value(), input_, input_.consumers())

	data = tf.placeholder(dtype=var_input.dtype, shape=var_input.shape)
	assign = tf.assign(var_input, data)

	score = tf.reduce_max(logits)
	index = tf.argmax(logits, axis=1)[0]

	# feed label based on index during session
	one_hot = tf.one_hot([index], 1001)
	label = tf.placeholder(dtype=tf.float32, shape=logits.shape)

	loss = tf.losses.softmax_cross_entropy(label, logits)

	entropy = - tf.reduce_sum(prob * tf.log(prob))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

	gv = optimizer.compute_gradients(loss, var_list=[var_input])[0]

	with tf.Session() as sess:
		for i in range(n_iter):
			# specify an image to perturb
			_ = sess.run(assign, {data: [a]})

			_one_hot = sess.run(one_hot)
			_g, _v = sess.run(gv, {label: _one_hot})

			print(sess.run(score), key[sess.run(index)], sess.run(entropy))

			# update
			_g = _g[0]
			_g[_g > 0] = 1.0 / 255
			_g[_g < 0] = -1.0 / 255
			a = a + epsilon * _g

		# final image
		_ = sess.run(assign, {data: [a]})
		print(sess.run(score), key[sess.run(index)], sess.run(entropy))

		# save the image	
		a_modified = sess.run(var_input)[0]
		a_modified[a_modified > 1] = 1
		a_modified[a_modified < -1] = -1
		a_out = resize(a_modified, list(original.shape)[:2], mode='constant')

		img = Image.fromarray((a_out * 255).astype('uint8'))
		img.save('out.png')

		pass



