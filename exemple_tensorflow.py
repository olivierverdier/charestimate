#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:58:23 2017

@author: bgris
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tl
#%% Graph
inp = tf.placeholder(shape=(None,10), dtype=tf.float32)
tf.contrib.image.transform
f_layer = tl.fully_connected(inp, num_outputs=4096)
reshaped = tf.reshape(f_layer, (-1, 4, 4, 256))
conv1 = tl.conv2d_transpose(reshaped, num_outputs=128, kernel_size=5, stride=2)
conv2 = tl.conv2d_transpose(conv1, num_outputs=64, kernel_size=5, stride=3)
output = tl.conv2d_transpose(conv2, num_outputs=2, kernel_size=5, stride=2)
#output = tf.contrib.layers.fully_connected(mid_layer, num_outputs=25*2, activation_fn=None)
vel = tf.placeholder(shape=(None, 48, 48, 2), dtype=tf.float32)
loss = tf.norm(vel - output)

#%% Session
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
output.eval(feed_dict={inp: np.ones((1,10))}).squeeze().shape
loss.eval(feed_dict={inp: np.ones((1,10)), vel: np.ones((1, 48, 48, 2))})
grads = tf.gradients(loss, [inp])
grad = grads[0]
grad.eval(feed_dict={inp: np.ones((1,10)), vel: np.ones((1, 48, 48, 2))})
optimizer = tf.train.AdamOptimizer()
mini_step = optimizer.minimize(loss)
session.run(tf.global_variables_initializer())
for i in range(100):
   session.run(mini_step, feed_dict={inp: np.ones((2,10)), vel: np.ones((2,48, 48, 2))})
#
output.eval(feed_dict={inp: np.ones((1,10))}).squeeze()

