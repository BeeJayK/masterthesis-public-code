#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:06:20 2021

@author: marlinberger

This is the first architecture written for the motion classification project.
It's purpose is to validate the input-pipeline for the data, check the
implementation of wandb etc.

The Model it self is a simple LSTM architecture. This is chosen, as it's an
architecture, that needs a stacked input, which is assumend to be the most
complex input.
"""
# python packages
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
global USE_WANDB, SWEEP, SAVE_LOCAL, PROJECT_NAME, SWEEP_MAX_ITTERS
# all vars here in [True, False], determine general modes
USE_WANDB = False
SWEEP = False
SAVE_LOCAL = False
# where to safe at wandb
PROJECT_NAME = "mt_first_real_data"
# how many itterations to perform at the sweep. choose =False for infinite
SWEEP_MAX_ITTERS = False


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
        "fps_reduce_fac": 3,
        "sequence_len": 2,
        "train_data_pack": "TrainPack_s1",
        "val_data_pack": "ValPack_s2",

        # hypers data
        "reduction_method": "manual_COM",

        # hypers net
        "LSTM_cells": 20,
        "LSTM_layers": 2,

        # hypers compile
        "learning_rate": 0.05,
        "optimizer": "adam",
        "epochs": 10,
        "batch_size": 3,

        # callback stuff and compile params, that are not planned to be
        # changed by a sweep
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "monitor_metric": "val_loss",
        "USE_EARLY_STOPPING": False,
        "early_stopping_patience": 10,
        "Re_LR_o_Pl": False,
        "reduce_lr_on_plateau_patiance": 4,
        "reduce_lr_on_plateau_factor": .5,

        # global user settings
        "USE_WANDB": USE_WANDB,
        "SWEEP": SWEEP,
        "SAVE_LOCAL": SAVE_LOCAL,
        "PROJECT_NAME": PROJECT_NAME,
        "SWEEP_MAX_ITTERS": SWEEP_MAX_ITTERS
    }
    # ----------------

    # possibly setup wandb connections and stuff
    run, config = neu_hlp.setup_wandb_run(config)
    # ensure the config is assigned rightly with the wandb config!
    print(config)

    # get the absolute location of this script, so the DataBuilder can access
    # the data stored locally
    abs_path = mt_hlp.get_abs_script_path(__file__)

    # build the dataset for the current run
    DS_Builder = neu_hlp.DataBuilder(config, abs_path)
    dataset_train = DS_Builder.build("train")
    dataset_val = DS_Builder.build("val")

    # debug/show dataset instances if desired
    neu_hlp.debug_ds_printer(dataset_train, debug_mode=True)

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
    model.add(cus_lay.Hand_Imputer(trainable=False))
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
        optimizer=config["optimizer"],
        loss=config["loss"],
        metrics=config["metrics"]
    )

    # give an overview of the current model
    print("\n", model.summary(), "\n")

    # initialise the wandb callback if wandb is used
    callbacks = neu_hlp.callback_initialiser(config, abs_path, run)

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
    # start a normal run
    # ----------------
    if not SWEEP:
        train_baseline()

    # start a sweep serach job
    # ----------------
    elif SWEEP and USE_WANDB:
        # configure the sweep params
        sweep_configs = {
            "method": "bayes",
            "metric": {
                "name": "val_loss",
                "goal": "minimize"
            },
            "parameters": {
                "fps_reduce_fac": {
                    "min": 1,
                    "max": 20
                },
                "LSTM_cells": {
                    "min": 1,
                    "max": 100
                },
                "LSTM_layers": {
                    "min": 1,
                    "max": 10
                },
                "optimizer": {
                    "values": ["adam", "RMSprop"]
                }
            }
        }
        # start the sweep
        neu_hlp.start_wandb_sweep(
            sweep_configs,
            PROJECT_NAME,
            train_baseline,
            max_iters=SWEEP_MAX_ITTERS
        )

    # error feedback for false initialisation of globals
    # ----------------
    else:
        print(
            "\nUps, choosen global settings wether start a run, nor a sweep\n"
        )
