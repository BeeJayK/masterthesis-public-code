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
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

# own pakages
from motion_tracking_helpers import coordinate_position_assignment as cpa
from motion_tracking_helpers import custom_layers as cus_lay
from motion_tracking_helpers import mt_neural_helpers as neu_hlp

import ExampleTensors as ExT

# get the dummy dataset
ds = neu_hlp.get_dev_ds()
# for elem in ds.take(10):
#	print(elem)

# determined in the get_dev_ds()-function
seq_len = 5
initial_dims = 3

# initialise the dev model
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(
    input_shape=(seq_len, initial_dims))
)
model.add(cus_lay.Normalizer(
    strategy="seq_on_last", debug_mode=True, trainable=False, dynamic=True)
)

# compile with meaningless config
model.compile()
model.fit(ds, epochs=1)
