#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:06:20 2021

@author: marlinberger

Test if a saved model can be loaded. This is mandatory and should be checked
after every new feature that is implemented in a model.
"""
# python packages
import os
from pathlib import Path
import shutil
import sys
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm

# own packages
from motion_tracking_helpers import motion_tracking_helpers as mt_hlp
from motion_tracking_helpers import mt_neural_helpers as neu_hlp
from motion_tracking_helpers import custom_layers as cus_lay

# USER INPUTS
# ----------------
# the name of the folder in 99_Architectures/_saves, in which the desired model
# lies
model_name = "2022-01-06--13-14-54--offline_run"
# ----------------

# get the path to all saved runs & models
paths_of_training_runs = mt_hlp.get_filepaths_in_dir(
    Path("99_Architectures/_saves")
)
# extract the relevant model's path
model_folder = [
    path for path in paths_of_training_runs if model_name in str(path)
]
# only continue if the assignment was unique
if len(model_folder) == 1:
    # extract the path to the h5 file (-> model)
    folder_cont = mt_hlp.get_filepaths_in_dir(model_folder[0])
    model_path = [path for path in folder_cont if "h5" in str(path)]
    # only continue if the assignment was unique
    if len(model_path) == 1:
        # load the model, collect all possible custom objects
        model = keras.models.load_model(model_path[0])
        # show model summary as control instance for a user
        print("\n\nSeems that everything workes fine\n\n")
        print(model.summary())
    # error print
    else:
        print("Error: None or more than one .h5 models found in model folder")
# error print
else:
    print("Error: None or more than one folders matching the given name")
