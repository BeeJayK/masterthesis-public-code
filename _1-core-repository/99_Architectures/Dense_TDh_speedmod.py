#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:06:20 2021

@author: marlinberger

The Model is a Dense architecture, with the Dense layers explicitly wrapped in
time-distributed layers. These shall learn independent hand features, while
hidden dense layers, after the tensors are flattened, shall learn the temporal
context, before the output layer makes a prediction. Preprocessing is integrated
with custom layers
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
from motion_tracking_helpers import mt_plot_helpers as plt_hlp

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
        "fps_reduce_fac": 1,
        "sequence_len": 30,
        "train_data_pack": "straight_80_20_train",
        "val_data_pack": "straight_80_20_val",
        "clip_max": False,

        # hypers data
        "normalize_strat": "inst_on_self",
        "reduce_strat": "_full",

        # hypers net
        "Dense_TD_cells": 400,
        "Dense_TD_layers": 15,
        "Dense_flat_cells": 200,
        "Dense_flat_layers": 20,
        "activation": "relu",

        # hypers compile
        "optimizer": "adam",
        "init_lr": 0.00001,
        "epochs": 1000,
        "batch_size": 32,

        # callback stuff and compile params, that are not planned to be
        # changed by a sweep
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "monitor_metric": "val_loss",
        "USE_EARLY_STOPPING": True,
        "early_stopping_patience": 25,
        "Re_LR_o_Pl": True,
        "reduce_lr_on_plateau_patiance": 7,
        "reduce_lr_on_plateau_factor": .1,
        "reduce_lr_on_plateau_min_delta": 0.001,
        "INHIBIT_CALC_BY_TIME_APPX": False,
        "INHIBIT_TIME_APPX_THRESH": 200,
        "INHIBIT_CALC_BY_MAX_LOOKBACK": True,
        "INHIBIT_MAX_LOOKBACK_THRESH": 3.5,
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

    # create a preprocessing model
    # ------------------
    model_pre = keras.models.Sequential()
    # use an explicit input layer to gain some freedom in the custom layers
    # before compiling
    model_pre.add(keras.layers.InputLayer(
        input_shape=(
            config["sequence_len"],
            config["initial_dimensions"]
        ))
    )
    # add custom input layers. this is basically the preprocessing pipeline
    model_pre.add(cus_lay.Hand_Input_Sorter(debug_mode=False, trainable=False))
    model_pre.add(cus_lay.Hand_Imputer(debug_mode=False, trainable=False))
    model_pre.add(cus_lay.Normalizer(
        strategy=config["normalize_strat"], debug_mode=False, trainable=False)
    )
    model_pre.add(cus_lay.Manual_Dim_Reducer(
        strategy=config["reduce_strat"], debug_mode=False, trainable=False)
    )
    # compile the preprocessing model
    model_pre.compile()

    # create the core model
    # ------------------
    model = keras.models.Sequential()
    # use an explicit input layer to gain some freedom in the custom layers
    # before compiling. reference on the preprocessing model
    model.add(keras.layers.InputLayer(
        input_shape=(
            config["sequence_len"],
            model_pre.layers[-1].output_shape[2]
        ))
    )
    # manage the Dense layers
    for layer_n in range(config["Dense_TD_layers"]):
        # NOTE: the standard dense-layer's behaviour on multidimensionl data
        #       is the same as when wrapped in a time-distributed layer, but
        #       to keep it clear what is happening, the dense layer is added
        #       wrapped
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    config["Dense_TD_cells"],
                    activation=config["activation"]
                )
            )
        )
    # get rid of one dimension
    model.add(keras.layers.Flatten())
    # possibly add hidden layers that might learn temporal pattern
    for _ in range(config["Dense_flat_layers"]):
        model.add(
            tf.keras.layers.Dense(
                config["Dense_flat_cells"],
                activation=config["activation"]
            )
        )
    # the output layer
    model.add(
        keras.layers.Dense(config["motion_classes"], activation="softmax")
    )
    # compile the model with the choosen configurations
    model.compile(
        optimizer=neu_hlp.configure_optimizer(config),
        loss=config["loss"],
        metrics=config["metrics"]
    )
    # ------------------

    # give an overview of the current model
    print("\n", model_pre.summary())
    print(model.summary(), "\n")

    # analyse the training that is just about to start. extract some final
    # information and append them to the config before saving everything
    # print them as little overview
    config = neu_hlp.wrapup_run_analysation(model, config)

    # initialise the callbacks, including the wandb callback if wandb is used
    # also the local version of the config-dict is possibly created here
    callbacks = neu_hlp.callback_initialiser(
        config, abs_path, run, add_prep_model=True, prep_model=model_pre
    )

    # possibly inhibit run's from training, if e.g. the combination of the
    # fps_reduction & seq_len, given by the sweep, would lead to extreme long
    # training times
    # possibly implement more checks here
    inhibit = neu_hlp.final_check_before_training(config)
    if inhibit:
        # possibly close wandb-connection and exit the training function
        neu_hlp.close_wandb_run(config, run, failed=True)
        return(False)
    
    # create the preprocessed datasets. do it after a possible inhibitation,
    # as this might take some time
    # build the dataset for the current run
    DS_Builder = neu_hlp.DataBuilder_v2(
        config, abs_path, clip_max=config["clip_max"]
    )
    dataset_train = DS_Builder.build("train")
    dataset_val = DS_Builder.build("val")
    # get the datasets preprocessed by the dedicated model for super-fast
    # training. the models are merged at every model's-saving-point
    print("Process the datasets by the preprocessing model for faster training")
    ds_prep_train = neu_hlp.create_preprocessed_dataset(
        model_pre, dataset_train
    )
    ds_prep_val = neu_hlp.create_preprocessed_dataset(
        model_pre, dataset_val
    )
    # debug/show dataset instances if desired
    neu_hlp.debug_ds_printer(dataset_train, debug_mode=False)
    neu_hlp.debug_ds_printer(ds_prep_train, debug_mode=False)

    # fit on the dataset
    model.fit(
        ds_prep_train,
        epochs=config["epochs"],
        validation_data=ds_prep_val,
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