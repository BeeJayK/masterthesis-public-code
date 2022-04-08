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
from motion_tracking_helpers import custom_layers as cus_lay

# WHATS THERE
# -----------
# base vec
subtract_vec_base = tf.constant(
    np.array([0 for _ in range(12)]),
    dtype=tf.float32
)

# wrist simu. NOTE: dtype NEEDS to be float! TODO: check if this is the case
val1 = tf.constant([1], dtype=tf.float32)
val2 = tf.constant([2], dtype=tf.float32)
val3 = tf.constant([3], dtype=tf.float32)

# dim simu
val_1_idx = np.array([1, 4, 7])
val_2_idx = np.array([2, 5, 8])
val_3_idx = np.array([3, 6, 9])

# validly-detected-hands-idx_s-in-batch simu
valid_hand_idx_s = tf.constant([[0], [4], [6], [7]], dtype=tf.int64)

instance = tf.constant(np.array(
    [[0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, ],
     [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], ]),
    dtype=tf.float32)


# WHATS TBD/implement
# ---------
def _update_scatter(val, idx_s, tensor):
    # expand dims to properly use the array with tf's scatter_nd function
    # TODO: das hier im self machen
    val_idx = tf.convert_to_tensor(np.expand_dims(idx_s, 1), dtype=tf.int32)

    # create repeated vec (which will be the coordinates of the wrist points)
    # whichs contains the coordinate the number of times it will be imputed
    # into the substraction vector
    rep_val = tf.repeat(val, repeats=tf.shape(val_1_idx)[0])

    # scattered inputs
    updated_tensor = tf.tensor_scatter_nd_update(
        tensor,
        val_idx,
        rep_val
    )
    return (updated_tensor)


# loop through the coordinate points. work with return values according to
# the autograph guidlines
subtract_vec = subtract_vec_base
# NOTE: zip wont word with tensorflow. loop needs to work with index-assignment
for val, idx_s in zip([val1, val2, val3], [val_1_idx, val_2_idx, val_3_idx]):
    subtract_vec = _update_scatter(val, idx_s, subtract_vec)

# now make the subtract_vec to the substract matrice, which's shape matches
# the seq-len + has zero'ed out vectors (->frame representations) where no valid
# hand got detected therefore first repeat the vector the times there are valid
# hands
rep_sub_vec = tf.repeat(
    tf.expand_dims(subtract_vec, axis=0),
    repeats=tf.shape(valid_hand_idx_s)[0],
    axis=0
)
# create the matrix with zero'ed out vals on unvalid hand positions
subtraction_matrix = tf.scatter_nd(
    # NOTE/TODO: checken wie die dimensionen im cus-layer aussehen
    valid_hand_idx_s,
    rep_sub_vec,
    # TODO: die müssen im cus-layer geholt werden
    shape=tf.constant([8, 12], dtype=tf.int64)
)

# make the final move: process the instance by subtracting the created matrix
# NOTE: in the real layer, this path needs to be taken two times - seperate for
# 		each hand - and then combined (create two matrices which are zeroed
# 		for the other hand, when both are computed, add them to the final
# 		matrix)
result = tf.math.subtract(instance, subtraction_matrix)

# SHOWCASE OUT
# ------------
print("\n")
print("base vec : ", subtract_vec_base)
print("val1     : ", val1)
print("idxs_1   : ", val_1_idx, type(val_1_idx))
print("sub base : ", subtract_vec)
print("val hands: ", valid_hand_idx_s)
print("sub matr : ", subtraction_matrix)

print("\n\n\n\n\nExample:\n")
print("inst     : ", instance)
print("finally  : ", result)
