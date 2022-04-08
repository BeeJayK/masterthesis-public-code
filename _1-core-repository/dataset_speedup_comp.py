#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:06:20 2021

@author: marlinberger

script to develop and analyse a significantly faster dataset-builder.
Use the infrastructure of the Model_Analyser-Routine
"""
# python packages
from re import S
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
import time
from tqdm import tqdm

# own pakages
from motion_tracking_helpers import motion_tracking_helpers as mt_hlp
from motion_tracking_helpers import mt_neural_helpers as neu_hlp
from motion_tracking_helpers import mt_plot_helpers as plt_hlp
# this import is necessary sothat the custom_layers of the loaded model can be
# decoded
from motion_tracking_helpers import custom_layers


# -----------------------------
analyse_folder = "speedup_checks"
analyse_datapack = ["Mixed_small"]#Nils_straight_80_20_val, Holdout_Heiko
# -----------------------------

# collect all information about the choosen evaluation folder set
Eval_Specs = mt_hlp.Eval_Specs(
    analyse_folder, MODE="Analyzer"
)

abs_path = (
    mt_hlp.get_abs_script_path(__file__) /
    mt_hlp.name_constants('ARCHITECTURE_DIR_NAME')
)

def old_method(model_config, abs_path):
    t = time.time()
    # build the dataset that's to be analyzed
    DS_Builder = neu_hlp.DataBuilder(model_config, abs_path)
    dataset_2_analyse = DS_Builder.build("analyse")
    print(f"building took {time.time()-t:.2f} s")
    return(dataset_2_analyse)

def new_method(model_config, abs_path):
    t = time.time()
    # build the dataset that's to be analyzed
    DS_Builder = neu_hlp.DataBuilder_v2(model_config, abs_path)
    dataset_2_analyse = DS_Builder.build("analyse")
    print(f"building took {time.time()-t:.2f} s")
    return(dataset_2_analyse)

def evaluate_ds_on_speed(ds):
    # get the batches for the tqdm-bar
    dataset_batches = tf.data.experimental.cardinality(ds).numpy()
    # in here, a value from every batch will be saved and the sum of this list
    # is used to quick cross check that the differen modes produce simular
    # results
    ensure_corr = []
    # dev-counter
    i = 0
    # set up the time to measure how long the process takes
    t1 = time.time()
    for x, y in tqdm(ds, total=dataset_batches, disable=False):
        # save a random value to later ensure both methods produce the same
        # results
        ensure_corr.append(x[0][0][10].numpy())
        # dev stuff
        """
        i += 1
        if i == 2:
            break
        """
    # stop the time, print the results
    t2 = time.time()
    acumu = np.nan_to_num(np.array(ensure_corr), nan=.3631)
    safe_num_1 = np.sum(acumu)
    safe_num_2 = np.sum(acumu[:int(len(ensure_corr)*.7)])
    safe_num_3 = np.sum(acumu[-int(len(ensure_corr)*.3):])
    print(f"\ntook {t2-t1:.2f} s\nsafe_num1: {safe_num_1}   safe_num2: " \
          f"{safe_num_2}   safe_num3: {safe_num_3}")

# itterate over the models
for run_path in Eval_Specs.TRAIN_RUN_PATHS:
    # initialise the Eval_Specs-object on the current run
    Eval_Specs.setup_on_run(run_path)
    # get the config dict from the run, which is needed for many thing, e.g.
    # preparing the datasets
    model_config = Eval_Specs.model_config
    # itterate over the datapacks
    for datapack_name in analyse_datapack:
        # add the data_pack to be analyed to the model's config dict
        model_config["analyse_data_pack"] = datapack_name

        # old_method
        ds = old_method(model_config, abs_path)
        evaluate_ds_on_speed(ds)
        
        # new_method
        ds = new_method(model_config, abs_path)
        evaluate_ds_on_speed(ds)