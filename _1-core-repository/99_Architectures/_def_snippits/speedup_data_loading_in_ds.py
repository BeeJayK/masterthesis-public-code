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

# own modules
from _def_snippits import ExampleTensors as ExT


"""
NOTE: This shows the implementation for the lookup of x. y needs to be
      lookuped in a similar fashion. Just the creatin in the init is slightly
      different and the dimension differ, but it's basically the very same

NOTE: even strided slices function exist. with this, one could rework the
       integration of the fps_reduction, eventhough i think this is absolutly
       unnececceray
"""

# THIS HAPPENS IN THE INIT OF THE DATASET BUILDER
# ---------------------
# example for the lookup, having 5 hand representations
arr = np.array((
    [76, 52, 74], [83, 13, 64], [91, 94, 33], [54, 74, 55], [22, 27, 25]
    )
)
# convert it to a tensor
t = tf.convert_to_tensor(arr, dtype=tf.float16)
# assign the value for the seq_len. first entry is the actual length, second
# entry means "do not slice across the second dimensino" -> take full represen-
# tation(s)
seq_len = tf.constant([2,-1])
# initialise here as this number is every call the same
start_at_0 = tf.constant([0])

# THIS HAPPEND AT EACH CALL FOR A NEW BATCH/INST
# ---------------------
# NOTE: in the real call, the number will already fly in as a tf.constant!
# first entry is the actual starting point, second entry means "grab the values
# from the representation starting at index 0", as this is the second axis
this_inst_start_idx = tf.constant([1])
slice_start = tf.concat((this_inst_start_idx, start_at_0), axis=0)
# slice out the instance
s =  tf.slice(
    t,
    slice_start,
    seq_len
)

# AND THATS ALREADY IT, NO NEED FOR PY_FUNC'S OR WHATEVER
# ---------------------
print("\n\nlookup tensor from init\n-----------------")
print(t)
print("\n\nseq_len-tensor from init\n-----------------")
print(seq_len)
print("\n\ninstances start idx, flying in every call\n-----------------")
print(this_inst_start_idx)
print("\n\nslice start\n-----------------")
print(slice_start)
print("\n\nsliced out instance, ready for the net\n-----------------")
print(s)
print("\n")