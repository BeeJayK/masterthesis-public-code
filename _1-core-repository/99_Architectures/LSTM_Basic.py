#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:06:20 2021

@author: marlinberger

The Model it self is a simple LSTM architecture, dynamically build during
sweeps, with variing properties. Preprocessing is integrated with custom layers
"""
# python packages
import logging
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf
import wandb

# own packages
from motion_tracking_helpers import motion_tracking_helpers as mt_hlp
from motion_tracking_helpers import mt_neural_helpers as neu_hlp
from motion_tracking_helpers import coordinate_position_assignment as cpa
from motion_tracking_helpers import custom_layers as cus_lay

# BASE USER CONFIGS
# ---------------
mt_hlp.initialise_logger(depth="info")
global USE_WANDB, SAVE_LOCAL, PROJECT_NAME
# all vars here in [True, False], determine general modes
USE_WANDB = True
SAVE_LOCAL = True
# where to safe at wandb
PROJECT_NAME = "MotionClassification_DCC"


# ---------------


def train_baseline():
    """write training in a function to make it triggerble from outer functions
    and enable search operations through parameter spaces

    Args:
        config (dict): a dictionary, that contains all needed variables to
                       run the model and document it

    Returns:
        run (wandb_obj): contains informations from wandb
    """
    # USER INPUTS
    # ----------------
    # set hypers etc. Mention everything in here, that shall be logged by wandb
    config = {
        # outline
        "Script": os.path.basename(__file__),
        "motion_classes": 10,
        "initial_dimensions": 133,
        "rnd_state": 42,
        "input_framerate": 29.72,

        # hypers data pre
        "fps_reduce_fac": 5,
        "sequence_len": 30,
        "train_data_pack": "straight_80_20_train",
        "val_data_pack": "straight_80_20_val",
        "clip_max": True,

        # hypers data
        "normalize_strat": "inst_on_self",
        "reduce_strat": "avg_of_dims_red",

        # hypers net
        "LSTM_cells": 16,
        "LSTM_layers": 6,

        # hypers compile
        "optimizer": "RMSprop",
        "init_lr": 0.05,
        "epochs": 20,
        "batch_size": 32,

        # callback stuff and compile params, that are not planned to be
        # changed by a sweep
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "monitor_metric": "val_loss",
        "USE_EARLY_STOPPING": True,
        "early_stopping_patience": 5,
        "Re_LR_o_Pl": True,
        "reduce_lr_on_plateau_patiance": 2,
        "reduce_lr_on_plateau_factor": .25,
        "reduce_lr_on_plateau_min_delta": 0.002,
        "INHIBIT_CALC_BY_TIME_APPX": False,
        "INHIBIT_TIME_APPX_THRESH": 200,
        "INHIBIT_CALC_BY_MAX_LOOKBACK": True,
        "INHIBIT_MAX_LOOKBACK_THRESH": 3.1,
        "INHIBIT_CALC_BY_MAX_PARAMS": True,
        "INHIBIT_MAX_PARAMS_THRESH": 15000000,

        # global user settings
        "USE_WANDB": USE_WANDB,
        "SAVE_LOCAL": SAVE_LOCAL,
        "PROJECT_NAME": PROJECT_NAME
    }
    # ----------------

    # possibly setup wandb connections and stuff
    run, config = neu_hlp.setup_wandb_run(config)
    # ensure the config is assigned rightly with the wandb config!
    print(config)

    # get the absolute location of this script, so the DataBuilder can access
    # the data stored locally, as well as the local saving with the best-model-
    # save-callback works
    abs_path = mt_hlp.get_abs_script_path(__file__)

    # build the dataset for the current run
    DS_Builder = neu_hlp.DataBuilder_v2(
        config, abs_path, clip_max=config["clip_max"]
    )
    dataset_train = DS_Builder.build("train")
    dataset_val = DS_Builder.build("val")

    # debug/show dataset instances if desired
    neu_hlp.debug_ds_printer(dataset_train, debug_mode=False)

    # create basic model
    # ------------------
    model = keras.models.Sequential()
    # use an explicit input layer to gain some freedom in the custom layers
    # before compiling
    model.add(keras.layers.InputLayer(
        input_shape=(
            config["sequence_len"],
            config["initial_dimensions"]))
    )

    # add custom input layers. this is basically the preprocessing pipeline
    model.add(cus_lay.Hand_Input_Sorter(debug_mode=False, trainable=False))
    model.add(cus_lay.Hand_Imputer(debug_mode=False, trainable=False))
    model.add(cus_lay.Normalizer(
        strategy=config["normalize_strat"], debug_mode=False, trainable=False)
    )
    model.add(cus_lay.Manual_Dim_Reducer(
        strategy=config["reduce_strat"], debug_mode=False, trainable=False)
    )

    # manage the (hidden) LSTM layers
    for layer_n in range(config["LSTM_layers"]):
        # return sequences for every but the last hidden layer
        return_sequence = (
            True if layer_n != config["LSTM_layers"] - 1 else False
        )
        # add the hidden layer
        model.add(keras.layers.LSTM(
            config["LSTM_cells"],
            return_sequences=return_sequence)
        )
    # the output layer
    model.add(
        keras.layers.Dense(config["motion_classes"], activation="softmax")
    )
    # ------------------

    # compile the model with the choosen configurations
    model.compile(
        optimizer=neu_hlp.configure_optimizer(config),
        loss=config["loss"],
        metrics=config["metrics"]
    )

    # give an overview of the current model
    print("\n", model.summary(), "\n")

    # analyse the training that is just about to start. extract some final
    # information and append them to the config before saving everything
    # print them as little overview
    config = neu_hlp.wrapup_run_analysation(model, config)

    # initialise the callbacks, including the wandb callback if wandb is used
    # also the local version of the config-dict is possibly created here
    callbacks = neu_hlp.callback_initialiser(config, abs_path, run)

    # possibly inhibit run's from training, if e.g. the combination of the
    # fps_reduction & seq_len, given by the sweep, would lead to extreme long
    # training times
    # possibly implement more checks here
    inhibit = neu_hlp.final_check_before_training(config)
    if inhibit:
        # possibly close wandb-connection and exit the training function
        neu_hlp.close_wandb_run(config, run, failed=True)
        return(False)

    # fit on the dataset
    model.fit(
        dataset_train,
        epochs=config["epochs"],
        validation_data=dataset_val,
        callbacks=callbacks
    )

    # make some predictions and show them if desired
    neu_hlp.debug_prediction(dataset_train, model, debug_mode=False)

    # close current run if wandb is used
    neu_hlp.close_wandb_run(config, run)


if __name__ == "__main__":
    # trigger the training function. sweeps will be started by agents, whichs
    # params are defined in yaml-files, wether local (saved in the github
    # repo) or directly through the wandb-UI
    train_baseline()