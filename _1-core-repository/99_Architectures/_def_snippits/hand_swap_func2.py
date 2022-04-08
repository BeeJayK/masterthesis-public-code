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

# csv input simulate
# ------------
X = tf.convert_to_tensor(
    np.array([0, 6, 6, 6, 0, 7, 7, 7, .906, .92], dtype=np.float32)
)

# im __init__()
# ------------
hs = tf.Variable([9, 8], dtype=tf.float32)
hs_prob = tf.Variable([0, 0], dtype=tf.float32)
nan_mask = tf.Variable([9, 8], dtype=tf.float32)
nan_count = tf.Variable(np.nan, dtype=tf.float32)
check_const_0 = tf.constant(0, dtype=tf.float32)
check_const_1 = tf.constant(1, dtype=tf.float32)
check_const_2 = tf.constant(2, dtype=tf.float32)
skip_eval = tf.Variable(np.array([True for _ in range(6)]), dtype=tf.bool)
# inputs from cpa
pos_0_idx = 0
pos_1_idx = 4
idxs_0 = np.array([1, 2, 3])
idxs_1 = np.array([5, 6, 7])
prob_0_idx = 8
prob_1_idx = 9
# swap pattern creation (TODO: put in own function)
ar = cpa.CSV_Architecture_v1()  # thats for the real layer
len_ar = len(ar.assignment)  # thats for the real layer
len_ar = len(X)
all_idxs = np.array([i for i in range(len_ar)])
all_idxs[idxs_0], all_idxs[idxs_1] = all_idxs[idxs_1], all_idxs[idxs_0]
swap_pattern = tf.constant(np.expand_dims(all_idxs, -1))


# dummy feedback functions
# ------------
def direct_return(): return ("direct_return")


def pass_cond(): return ("pass")


# every call() calculate these
# ------------
h_0 = X[pos_0_idx]
h_1 = X[pos_1_idx]
hs.assign([h_0, h_1])
h_0_prob = X[prob_0_idx]
h_1_prob = X[prob_1_idx]
hs_prob.assign([h_0_prob, h_1_prob])
nan_mask.assign(
    tf.dtypes.cast(
        tf.math.is_nan(hs),
        dtype=tf.float32)
)
nan_count = tf.math.reduce_sum(nan_mask)


# TODO: gucken dass nicht unnötig gerechnet wird: was muss wirklich jeden
#	    call gemacht werden und was besser in den funktionen?!

# condition section
# -------------
def skip1(hs):
    return (tf.greater(hs[1], hs[0]))


def skip2(nan_count, check_const_2):
    return (tf.equal(nan_count, check_const_2))


def skip3(hs, nan_mask, check_const_1):
    h_0_is_nan = tf.equal(nan_mask[0], check_const_1)
    h_1_is_1 = tf.equal(hs[1], check_const_1)
    return (tf.math.logical_and(h_0_is_nan, h_1_is_1))


def skip4(hs, nan_mask, check_const_0, check_const_1):
    h_1_is_nan = tf.equal(nan_mask[1], check_const_1)
    h_0_is_0 = tf.equal(hs[0], check_const_0)
    return (tf.math.logical_and(h_1_is_nan, h_0_is_0))


def skip5(hs, hs_prob, nan_count, check_const_0):
    no_nan = tf.equal(nan_count, check_const_0)
    h_0_equals_h_1 = tf.equal(hs[0], hs[1])
    h_n_equals_0 = tf.equal(hs[0], check_const_0)
    h_0_p_greater = tf.greater(hs_prob[0], hs_prob[1])
    final_eval = tf.math.logical_and(
        tf.math.logical_and(no_nan, h_0_equals_h_1),
        tf.math.logical_and(h_n_equals_0, h_0_p_greater)
    )
    return (final_eval)


def skip6(hs, hs_prob, nan_count, check_const_0, check_const_1):
    no_nan = tf.equal(nan_count, check_const_0)
    h_0_equals_h_1 = tf.equal(hs[0], hs[1])
    h_n_equals_1 = tf.equal(hs[0], check_const_1)
    h_1_p_greater = tf.greater(hs_prob[1], hs_prob[0])
    final_eval = tf.math.logical_and(
        tf.math.logical_and(no_nan, h_0_equals_h_1),
        tf.math.logical_and(h_n_equals_1, h_1_p_greater)
    )
    return (final_eval)


print(skip_eval)
# SKIP 1
skip_eval[0].assign(
    skip1(hs)
)

# SKIP 2
skip_eval[1].assign(
    skip2(nan_count, check_const_2)
)

# SKIP 3
skip_eval[2].assign(
    skip3(hs, nan_mask, check_const_1)
)

# SKIP 4
skip_eval[3].assign(
    skip4(hs, nan_mask, check_const_0, check_const_1)
)

# SKIP 5
skip_eval[4].assign(
    skip5(hs, hs_prob, nan_count, check_const_0)
)

# SKIP 6
skip_eval[5].assign(
    skip6(hs, hs_prob, nan_count, check_const_0, check_const_1)
)

print(skip_eval)


# pass func
def pass_func():
    return (X)


# swap function
def swap():
    shape = tf.shape(X, out_type=tf.int64)
    scattered_tensor = tf.scatter_nd(swap_pattern, X, shape)
    return (scattered_tensor)


# if in the real layer, it went until here: swap the handcoordinates
print(X)
X = tf.cond(
    tf.math.reduce_any(skip_eval),
    pass_func,
    swap
)
print(X)

# NOTE
# condiciton overview
# build it in a way, that 'clean' samples get returned and if the function
# gets to a certain point, a swap will be triggered
# ---------------------
# if True, direct return:
# 	- keines nan, h_1 > h_0
# 	- beides nan
# 	- h_0 == nan, h_1 == 1
# 	- h_1 == nan, h_0 == 0
# 	- keines nan, h_0 == h_1, h_n == 0, h_0_p > h_1_p
# 	- keines nan, h_0 == h_1, h_n == 1, h_0_p < h_1_p
# else:
#	  swap()
# ---------------------
# this approach has one more condition than a check against swap, but
# it can also be exited as soon as one condition is true, which should be the
# case most of the times, so it shall be wayyyy faster


# NOTE
# swap mit tf.scatter_nd oder ähnlich
