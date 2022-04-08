#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:06:20 2021

@author: marlinberger

Convert h5 models to tflite, which is the desired format for the framework.
Note that this can get shitty for some kind of tf-layers.
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
model_name = "2022-01-04--13-54-57--offline_run"
# ----------------

"""
HIER GUCKEN WENN DIE SACHEN ZU TFLITE CONVERTIERT WERDEN SOLLEN.
EINFACHER IST ABER WAHRSCHEINLICH DIE CUSTOM LAYER DANN AUS EINEM MODEL ZU
ZIEHEN, DAS PREPROCESSING VORZULAGERN UND DAS KERN MODEL - FALLS DIESES
ÜBERHAUPT KONVERTIERBAR IST, DAS IST NÄMLICH AUCH NICHT FÜR ALLE TF FUNKTIONEN
GEGEBEN - IN TFLITE ZU KONVERTIEREN
https://www.tensorflow.org/lite/guide/ops_custom?hl=en
"""
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
        print(model.summary())
        # initialise the tflite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.allow_custom_ops = True
        # convert the model and save it in the same folder where the h5 model
        # lies
        tflite_model = converter.convert()
        open(f"{str(model_folder[0])}/model.tflite", "wb").write(tflite_model)
    # error print
    else:
        print("Error: None or more than one .h5 models found in model folder")
# error print
else:
    print("Error: None or more than one folders matching the given name")
