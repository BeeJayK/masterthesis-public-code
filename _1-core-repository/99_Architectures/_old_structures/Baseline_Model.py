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
from tensorflow import keras
import tensorflow as tf

# own packages
from motion_tracking_helpers import motion_tracking_helpers as mt_hlp
from motion_tracking_helpers import mt_neural_helpers as neu_hlp
from motion_tracking_helpers import coordinate_position_assignment as cpa
from motion_tracking_helpers import custom_layers as cus_lay

# USER INPUTS
# ----------------
# set hypers etc. Mention everything in here, that shall be logged by wandb
global config
config = {
    # outline
    "Model": "Baseline_Model",
    "motion_classes": 10,
    "initial_dimensions": 133,
    "rnd_state": 42,
    "input_framerate": 29.72,

    # hypers data pre
    "fps_reduce_fac": 3,
    "sequence_len": 2,
    "train_data_pack": "Trainpack",
    "val_data_pack": "Valpack",

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
    "SAVE_LOCAL": True,
    "USE_WANDB": False
}


# ----------------


def train(config):
    """write training in a function to make it triggerble from outer functions
    and enable search operations through parameter spaces

    Args:
        config (dict): a dictionary, that contains all needed variables to
                       run the model and document it

    Returns:
        run (wandb_obj): contains informations from wandb
    """
    # possibly open wandb connection
    # NOTE: CHECK IF THIS FUNCTION WORKS WITH SWEEP
    run = neu_hlp.setup_wandb(config)

    # get the absolute location of this script, so the DataBuilder can access
    # the data stored locally
    abs_path = mt_hlp.get_abs_script_path(__file__)

    # build the dataset for the current run
    DS_Builder = neu_hlp.DataBuilder(config, abs_path)
    dataset_train = DS_Builder.build("train")
    dataset_val = DS_Builder.build("val")

    # for debugging the dataset
    # NOTE: this is outsourced to the mt_neural_helpers script
    for inst in dataset_train.take(1):
        # for num in inst[0].numpy()[1][0]:
        # 	print(f"{num},")
        # print(inst[0])
        # print(inst[1])
        # print(inst[0].shape)
        # print(inst[1].shape)
        pass

    # TODO: add sweeps! -> check out Baseline_Sweep.py Script

    # TODO: check time differences with custom input layers. if to big,
    #		use different strategy

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

    # add custom input layers. possibly bundle them. preprocessing pipeline 2
    model.add(cus_lay.Hand_Input_Sorter(debug_mode=False, trainable=False))
    # model.add(cus_lay.Manual_Dim_Reducer())
    model.add(cus_lay.Hand_Imputer(trainable=False))
    # model.add(cus_lay.NaN_to_zero())
    # model.add(cus_lay.Normalizer())

    # manage the (hidden) LSTM layers
    for layer_n in range(config["LSTM_layers"]):
        # return sequences for every but the last hidden layer
        return_sequence = (
            True if layer_n != config["LSTM_layers"] - 1 else False
        )
        # debug print
        print(f"return_sequence = {return_sequence}")
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

    # make some predictions, for debugging...
    # NOTE: this is outsourced to the mt_neural_helpers script
    pred_n = 3
    round_pred_to = 2
    for testinst_pre, testlabel_pre in dataset_train.take(pred_n):
        # keep only one instance of the given batch...
        testinst, testlabel = testinst_pre[0, :, :], testlabel_pre[0, :]
        # make prediction
        prediction = np.round(
            model.predict(np.expand_dims(testinst, 0)),
            round_pred_to)[0]
        # print it
        print("\n")
        print(f"prediction: {prediction}")
        print(f"groud_tru : {testlabel}")

    return (run)


if __name__ == "__main__":
    # write script in a triggerble form, so that it can be used along with
    # wandb's sweeü to optimize and search the hyperparam and datamanipulation
    # space
    run = train(config)

    # possibly close wandb connection
    run = neu_hlp.close_wandb(config, run)
