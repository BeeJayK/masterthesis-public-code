#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:06:20 2021

@author: marlinberger

Testscript to test tf functions and basic concepts to implement in custom
layers and stuff
"""
# python packages
from re import S
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

# own pakages
from motion_tracking_helpers import coordinate_position_assignment as cpa

# starting point
X = tf.constant(np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), dtype=tf.float32)
# simulate the different idx-lists for dimensions, intro vals, etc.
s0_idxs = np.array([0, 1])
s1_idxs = np.array([2, 4, 6, 8])
s2_idxs = np.array([3, 5, 7, 9])
print("\nstarting point:")
print(X)

# gather the tensor slices on themselve. in the custom layer, they now might
# be processed and changed
s0 = tf.gather(
    X,
    s0_idxs.tolist(),
    axis=-1
)
s1 = tf.gather(
    X,
    s1_idxs.tolist(),
    axis=-1
)
s2 = tf.gather(
    X,
    s2_idxs.tolist(),
    axis=-1
)

# concetenate the possibly modified slices as they are
X = tf.concat(
    [
        s0,
        s1,
        s2
    ],
    0
)
print("\nafter re-concetination after splitting by dims")
print(X)

# create the swap pattern to resort the dimensions into coordinate points, as
# they are delivered
swap_pattern_np = np.concatenate([s0_idxs, s1_idxs, s2_idxs])
swap_pattern = tf.constant(np.expand_dims(swap_pattern_np, -1))
# swap the tensor accoring to the pattern
shape = tf.shape(X, out_type=tf.int64)
X = tf.scatter_nd(swap_pattern, X, shape)
print("\ntensor after scattering with swap pattern")
print(X)
