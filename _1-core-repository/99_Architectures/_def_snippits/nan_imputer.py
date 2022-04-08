#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:06:20 2021

@author: marlinberger

Testscript to test tf functions and basic concepts to implement in custom
layers and stuff
"""
# python packages
import numpy as np
from tensorflow import keras
import tensorflow as tf

# own pakages
from motion_tracking_helpers import coordinate_position_assignment as cpa

input_array = np.array([2, np.nan], dtype=np.float32)
imputer = tf.constant(5.)
print(input_array)
print(imputer)

nan_mask = tf.math.is_nan(input_array)
print(nan_mask)

mask = tf.dtypes.cast(
    tf.math.logical_not(nan_mask),
    dtype=tf.float32
)
print(mask)

zeroed = tf.math.multiply_no_nan(input_array, mask)
print(zeroed)

inverse_mask = tf.dtypes.cast(
    nan_mask,
    dtype=tf.float32
)
print(inverse_mask)

add_mask = tf.math.scalar_mul(imputer, inverse_mask)
print(add_mask)

result = tf.add(zeroed, add_mask)
print(result)
