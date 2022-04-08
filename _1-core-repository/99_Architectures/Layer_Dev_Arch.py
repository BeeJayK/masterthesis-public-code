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
import logging
import os

# third party packages
from tensorflow import keras

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
USE_WANDB = False
SAVE_LOCAL = False
# where to safe at wandb
PROJECT_NAME = "motion_classf_tests"


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
        "fps_reduce_fac": 8,
        "sequence_len": 70,
        "train_data_pack": "Dev_Pack_mid_train",
        "val_data_pack": "Dev_Pack_mid_val",

        # hypers data
        "reduction_method": "manual_COM",

        # hypers net
        "LSTM_cells": 10,
        "LSTM_layers": 5,

        # hypers compile
        "optimizer": "RMSprop",
        "init_lr": 0.005,
        "epochs": 10,
        "batch_size": 32,

        # callback stuff and compile params, that are not planned to be
        # changed by a sweep
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "monitor_metric": "val_loss",
        "USE_EARLY_STOPPING": True,
        "early_stopping_patience": 10,
        "Re_LR_o_Pl": True,
        "reduce_lr_on_plateau_patiance": 2,
        "reduce_lr_on_plateau_factor": .25,
        "reduce_lr_on_plateau_min_delta": 0.002,
        "INHIBIT_CALC_BY_THRESH": True,
        "INHIBIT_THRESH": 80,

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
    DS_Builder = neu_hlp.DataBuilder_v2(config, abs_path)
    dataset_train = DS_Builder.build("train")
    dataset_val = DS_Builder.build("val")

    # debug/show dataset instances if desired
    neu_hlp.debug_ds_printer(dataset_train, debug_mode=False)
    neu_hlp.debug_ds_printer(dataset_val, debug_mode=False)

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
        strategy="seq_on_last", debug_mode=False, trainable=False)
    )

    #model.add(cus_lay.Passer(
    #    print_arg="INITIAL", trainable=False, name="Passer1")
    #)

    # the layer that's currently developed
    model.add(cus_lay.Manual_Dim_Reducer(
        strategy="avg_of_dims_red", debug_mode=False, trainable=False)
    )

    #model.add(cus_lay.Passer(
    #    print_arg="AFTER", trainable=False, name="Passer2")
    #)

    # TODO: maybe a final 0-1 normalizer in any form is needed -> CHECK THAT!

    # basic LSTM network
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
        metrics=config["metrics"],
        # run_eagerly=True
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
