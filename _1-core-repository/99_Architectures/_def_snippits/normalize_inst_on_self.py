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

# numerical-zero-handler-var
eps = 1e-8

# simulate the instance as it looks at the time it shall be processed
inst = tf.constant(np.array(
    [0.07812141, 0.09163176, 0.09641974, 0.09690223, 0.09493342
     ]), dtype=tf.float32)

inst2 = tf.constant(np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9]), dtype=tf.float32)

inst3 = tf.constant(np.array(
    [0, 0, 0, 0, 0, 0, 0, 0]), dtype=tf.float32)

inst4 = tf.constant(np.array(
    [2, 2, 2, 2, 2, 2, 2, 2]), dtype=tf.float32)


# normalize min/max to 0/1
def _min_max_0_1(X):
    """normalize each dimension and hand on itself
    """
    X = tf.math.divide(
        tf.math.subtract(
            X,
            tf.math.reduce_min(X)
        ),
        tf.add(
            tf.math.subtract(
                tf.math.reduce_max(X),
                tf.math.reduce_min(X)
            ),
            eps
        )
    )
    return (X)


print("\n")
print(inst)
print(_min_max_0_1(inst))
print("\n")
print(inst2)
print(_min_max_0_1(inst2))
print("\n")
print(inst3)
print(_min_max_0_1(inst3))
print("\n")
print(inst4)
print(_min_max_0_1(inst4))
