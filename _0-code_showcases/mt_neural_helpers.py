#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:03:40 2021

@author: marlinberger

helper functions in the context of the developing neural networks
"""

# python packages
import csv
from datetime import datetime
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sklearn as sk
import sys
import tensorflow as tf
from tqdm import tqdm
import wandb
from wandb.keras import WandbCallback

# modules from this package
from . import custom_layers as cus_lay
from . import motion_tracking_helpers as mt_hlp


def check_import():
    """use this function to check if the import worked
    """
    logging.debug("\nyou made it in\n")


def setup_wandb_run(config):
    """set up the wandb connection, based on the config file

    Args:
        config (dict): contains all informations about the current run

    Return:
        run (wandb_obj): contains informations from wandb
    """
    # determine wether wandb shall be used or not
    if config["USE_WANDB"]:
        os.environ["WANDB_MODE"] = "run"
        # login to wandb, use it to keep track of the trainings
        wandb.login()
        # initialise wandb
        run = wandb.init(
            project=config["PROJECT_NAME"],
            config=config
        )
        # this is mandatory if sweeps are used. assign the current version of
        # the wandb.config to the local config dict. If just a normal run is
        # done and not a sweep, this shouldnt change anything anyway
        config = dict(wandb.config)
    else:
        run = False
    return (run, config)


def start_wandb_sweep(sweep_configs, PROJECT_NAME, train_function,
                      max_iters=False):
    """wrapper for a training function that is already adapted for wandb's
    run structure. perform a sweep over it

    DEPRECATION INFO:
        switched mainly to the use of yaml-file, as this is less vulnerable
        when working with agents spreaded across several machines/processes
        for enhanced sweep-performance. Nevertheless: This function can still
        be used to initialise single-process sweeps on a given machine. This
        saves one from needing yaml initialiser files

    Args:
        sweep_configs (dict): contains all informations for the sweep
        PROJECT_NAME (str): in which wandb project this sweep will go
        train_function (func): the wandb compatible training function
        max_iters (False || int): if given, this determines the max runs done
    """
    # initialise the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configs,
        project=PROJECT_NAME
    )
    # start the agent with respect to a max number of itterations if given
    if max_iters:
        print("\n\nwith count\n\n")
        wandb.agent(sweep_id=sweep_id, function=train_function, count=max_iters)
    else:
        print("\n\nwithout count\n\n")
        wandb.agent(sweep_id=sweep_id, function=train_function)


def close_wandb_run(config, run, failed=False):
    """close the connections, signal that a run/sweep is done

    Args:
        config (dict): contains all informations about the current run
        run (wandb_obj): contains informations from wandb
    """
    if config["USE_WANDB"]:
        if not failed:
            # signal to wandb that the run is over
            run.finish()
        else:
            # mark the run as failed
            wandb.finish(exit_code=2)


def configure_optimizer(config):
    """set up the optimizer to use for training. It is done by this helper
    function, as the optimizer can only be used with it's default params if it
    is passed by the string-name directly to the model.compile-method.
    Therefore, parameterise each optimizer with the desired configs here and
    return the initialised optimizer object.

    Args:
        config (dict): contains all informations about the current run
    
    Returns:
        Optimizer (tf.keras.optimizers.Optimizer): a keras optimizer
    """
    # initialise the optimizers by given string
    if config["optimizer"] == "adam":
        Optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["init_lr"]
        )
    elif config["optimizer"] == "RMSprop":
        Optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=config["init_lr"]
        )
    return(Optimizer)


def callback_initialiser(config, abs_path, run):
    """initialise the callbacks on a central place. None of the param dependent
    variables are designed to be changed. they are just initialised along with
    all the other params, on a central place, in the config dict. if local
    saving is activated, also a local csv-version of this dict is created here

    Args:
        config (dict): a dictionary, that contains all needed variables to
                       run the model and document it
        abs_path (PosixPath): path of the triggering script. This is needed
                              to save the model locally
        run (wandb_obj): contains informations from wandb

    Returns:
        callbacks (list): customized callbacks to use with any model
    """
    # initialise the callbacks
    callbacks = []

    # add early stopping callback if desired
    if config["USE_EARLY_STOPPING"]:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=config["monitor_metric"],
                verbose=1,
                patience=config["early_stopping_patience"]
            )
        )

    # add learning-rate-reducer-on-plateau if desired
    if config["Re_LR_o_Pl"]:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=config["monitor_metric"],
                patience=config["reduce_lr_on_plateau_patiance"],
                verbose=1,
                factor=config["reduce_lr_on_plateau_factor"],
                min_delta=config["reduce_lr_on_plateau_min_delta"]
            )
        )

    # add wandb callback if it is to be used, to save all the important metrics
    # etc. exclude the model itself as this will take wayyyy to much space for
    # all the runs
    if config["USE_WANDB"]:
        callbacks.append(WandbCallback(save_model=False))

    # configure everything for the local save if desired, save the config file
    # there, point the callback there. Initialise the model_save_path to be
    # able to pass it back, no matter if it's assigend or not
    model_save_path = None
    if config["SAVE_LOCAL"]:
        # create directory to store best model at. Also save the config dict
        # there name of the folder is the current date
        save_time = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
        run_name = run.name if config["USE_WANDB"] else "offline_run"
        model_save_path = Path(
            f"{abs_path}/" \
            f"{mt_hlp.name_constants('MODEL_SAVE_DIR_NAME')}/" \
            f"{save_time}--{run_name}"
        )
        # make the dir, throw an error if it already exists (which should NEVER
        # happen)
        os.makedirs(
            str(model_save_path)
        )
        # write the config file in a readable form
        CONFIG_FILE_NAME = mt_hlp.name_constants("CONFIG_FILE_NAME")
        with open(
            str(model_save_path) + "/" + CONFIG_FILE_NAME, 'w'
        ) as csv_file:
            writer = csv.writer(csv_file)
            for key, value in config.items():
                writer.writerow([key, value])

        # add the best-model-callback
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=(str(model_save_path) + "/best_model.h5"),
                monitor=config["monitor_metric"],
                verbose=0,
                save_best_only=True
            )
        )

    return (callbacks)


def wrapup_run_analysation(model, config, verbose=True):
    """make some analysation of a parameterised model, the instances etc.
    Append the investigations to the config dict, possible update it on wandb

    Args:
        model (tf.model): the compiled model, ready to be fitted
        config (dict): contains all informations about the current run
    
    Returns:
        config (dict): the enlarged config
    """

    # indicating number how long the training will take
    # this indicator possibly needs to be calculated differently for different
    # architectures. But it's also possible that just the treshold for a inhibit
    # needs to be changed but the indicator stays valid.
    # NOTE: Check this when different architectures are available
    training_time_indicator = (
        (1./config["fps_reduce_fac"])
         * config["sequence_len"]
         * config["epochs"]
    )
    # how many parameter the whole model uses
    parameter_n = model.count_params()
    # how many seconds a model looks back for a prediction
    lookback_seconds = (
        (config["fps_reduce_fac"] / (config["input_framerate"]))
         * config["sequence_len"]
    )

    # collect all values, that shall appear in the run's config, no matter if
    # local, online or for further use in the architecture script
    update_entries = {
        "training_time_indicator": round(training_time_indicator, 2),
        "parameter_n": parameter_n,
        "lookback_seconds": round(lookback_seconds, 1)
    }

    # update the local config dictionary
    config.update(update_entries)
    # if wandb is enabled, also update the only config
    if config["USE_WANDB"]:
        wandb.config.update(update_entries)
    
    # show the extracted informations in the terminal before the training
    # starts to give a short overview
    if verbose:
        [print(key, val) for key, val in update_entries.items()]
        print("")

    return(config)


def final_check_before_training(config):
    """perform some final checks before the training is started, to possibly
    prevent the run from starting, e.g. if a combination of fps_red&seq_len
    would lead to extraordinary training times.
    Note that these checks are not necessarily performed, but only if desired
    (which is configured in the config file)

    Args:
        config (dict): contains all informations about the current run
    
    Retuns
        inhibit (Bool): if a run should not be started (==True) or everything
                        is fine
    """
    # dont inhibit the run if none of the following conditions triggers
    inhibit=False

    # if activated, inhibit a run if the THRESH for the training_time_indicator
    # is exeeded
    if config["INHIBIT_CALC_BY_TIME_APPX"] and (
            config["training_time_indicator"] > 
            config["INHIBIT_TIME_APPX_THRESH"]):
        inhibit=True
        print(
            "\nNOTE:\ninhibiting the run due to the\n" \
            "training_time_indicator exeeding\n" \
            "it's given threshold:\n"
            f"  INHIBIT_CALC_BY_TIME_APPX:  " \
                    f"{config['INHIBIT_CALC_BY_TIME_APPX']}\n" \
            f"  training_time_indicator:    " \
                    f"{config['training_time_indicator']}\n" \
            f"  INHIBIT_TIME_APPX_THRESH:   " \
                    f"{config['INHIBIT_TIME_APPX_THRESH']}\n\n" \
        )
    
    # if activated, inhibit a run if the THRESH for the training_time_indicator
    # is exeeded
    if config["INHIBIT_CALC_BY_MAX_LOOKBACK"] and (
            config["lookback_seconds"] > 
            config["INHIBIT_MAX_LOOKBACK_THRESH"]):
        inhibit=True
        print(
            "\nNOTE:\ninhibiting the run due to the\n" \
            "lookback_seconds exeeding\n" \
            "it's given threshold:\n"
            f"  INHIBIT_CALC_BY_MAX_LOOKBACK:  " \
                    f"{config['INHIBIT_CALC_BY_MAX_LOOKBACK']}\n" \
            f"  lookback_seconds:              " \
                    f"{config['lookback_seconds']}\n" \
            f"  INHIBIT_MAX_LOOKBACK_THRESH:   " \
                    f"{config['INHIBIT_MAX_LOOKBACK_THRESH']}\n\n" \
        )
    
    # if activated, inhibit a run if the THRESH for the maximum of model
    # parameters triggers
    if config["INHIBIT_CALC_BY_MAX_PARAMS"] and (
            config["parameter_n"] > 
            config["INHIBIT_MAX_PARAMS_THRESH"]):
        inhibit=True
        print(
            "\nNOTE:\ninhibiting the run due to the\n" \
            "model's parameter_n exeeding\n" \
            "theire given threshold:\n"
            f"  INHIBIT_CALC_BY_MAX_PARAMS:  " \
                    f"{config['INHIBIT_CALC_BY_MAX_PARAMS']}\n" \
            f"  parameter_n:                 " \
                    f"{config['parameter_n']}\n" \
            f"  INHIBIT_MAX_PARAMS_THRESH:   " \
                    f"{config['INHIBIT_MAX_PARAMS_THRESH']}\n\n" \
        )
    
    return(inhibit)


def debug_ds_printer(dataset, debug_mode=False):
    """print some information about the returned instances from a dataset

    Args:
        dataset (tf.dataset): the dataset to inspect
        dubug_mode (bool): wether to print something or not
    """
    if debug_mode:
        for inst in dataset.take(1):
            print(inst[0])
            print(inst[1])
            print(inst[0].shape)
            print(inst[1].shape)
        # some more custom stuff to play around during implementation of
        # new features
        # for num in inst[0].numpy()[1][0]:
        # 	print(f"{num},")


def debug_prediction(dataset, model, pred_n=3, debug_mode=False):
    """make some predictions after a training run, for debugging or showcase
    purposes

    Args:
        dataset (tf.dataset): dataset from which instances are taken for the
                              predictions
        model (tf.model): a trained model
        pred_n (int): how many predictions shall be done
        dubug_mode (bool): wether to do something or not

    """
    if debug_mode:
        # precision of the prints
        round_pred_to = 2
        # itterate through the desired number of instances
        for testinst_pre, testlabel_pre in dataset.take(pred_n):
            # keep only one instance of the given batch
            testinst, testlabel = testinst_pre[0, :, :], testlabel_pre[0, :]
            # make prediction
            prediction = np.round(
                model.predict(np.expand_dims(testinst, 0)),
                round_pred_to)[0]
            # print it
            print("\n")
            print(f"prediction: {prediction}")
            print(f"groud_tru : {testlabel}")


class DataBuilder():
    def __init__(self, config, abs_path, is_train=True, clip_max=False):
        """set initial vars, that are nice to grab over and
        over again and that are not to specific

        Args:
            config (dict): a dictionary, that contains all needed variables to
                           run the model and document it
            abs_path (PosixPath): path of the triggering script. This is needed
                                  to access the data stored locally
            is_train (bool): if we are in training mode or not
            clip_max (bool): if instances shall be reduces (-> clipped) when
                             their volume (num_n*seq_len) overcome a certain
                             threshold
        """
        # assign everything needed
        self.config = config
        # TODO: abs_path umbenennen in etwas was klarer ist: hier muss irgendwie
        #       der path vom 99_Architectures Ordner reingegeben werden
        self.abs_path = abs_path
        self.is_train = is_train
        # calculate whats needed and is depended of the inputs
        self.MIN_PICS_IN_DIR = self.get_needed_pic_for_run()
        # set everything up if clipping is involved
        self.clip_max = clip_max
        # for LSTM_Basic.py script, if the clipping value is reached, the
        # training will take about 30 hours @20_epochs
        self.clip_data_vol_thresh = 2e6 # num_n*seq_len
        # approximate the ration between train and val data amount, to clip the
        # val data by this reduced amount
        self.clip_train_val_ratio = .2

    def build(self, mode):
        """paths to files to scan

        Args:
            mode (str): in ["train", "val", "analyse"], determines which data
                        will be the base for the dataset that is returned

        Returns:
            ds (tf.Dataset): trainable/predictable dataset
        """
        # get the upper folderpaths
        if mode == "train":
            data_packname = self.config["train_data_pack"]
        elif mode == "val":
            data_packname = self.config["val_data_pack"]
        elif mode == "analyse":
            data_packname = self.config["analyse_data_pack"]
        folderpaths = self.get_datapack_folderpaths(data_packname)
        # check if all folderpath contain enough datapoints for the stacking
        folderpaths = self.identify_unvalid_folder(folderpaths)

        # get the stacked filepaths for every video
        filepaths = [mt_hlp.get_filepaths_in_dir(folderpath) for folderpath in
                     folderpaths]
        # sort paths ascending by the assigned framenumber to ensure that the
        # data is consistent
        filepaths = [sorted(paths, key=lambda x: int(x.name.split("_")[0])) for
                     paths in filepaths]
        # stack the paths according to the desired sequential input shape and
        # the desired fps reduction. Also convert the PosixPaths to strings on
        # the fly
        stacked_filpaths = self.stack_paths(filepaths)

        # shuffle the stacked paths at this point, so that no shuffling needs
        # to be performed during prefetching, which would be quiet a pain with
        # regard to the low number of input videos that are especially long
        shuffled_stacked_filpaths = sk.utils.shuffle(
            stacked_filpaths,
            random_state=self.config["rnd_state"])
        
        # if clipping is activated, possibly take an appropriate number n of
        # the first elements to reduce the dataset to a maximum fixed volume. As
        # the shuffeling of the paths happen before, this easy approach is very
        # efficient and valid
        shuffled_stacked_clipped_filpaths = self.redecue_if_clip(
            shuffled_stacked_filpaths, mode
        )
        
        # create the dataset from the shuffled stacked paths
        ds = tf.data.Dataset.from_tensor_slices(
            shuffled_stacked_clipped_filpaths
        )
        # load files and stack them
        ds = ds.map(self.load_vector_representations)

        # batch it and make it buffer automaticly
        ds = ds.batch(self.config["batch_size"])
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return (ds)

    def get_needed_pic_for_run(self):
        """check how many pictures are needed at least for one video based on
        the params in the config file

        Returns:
            MIN_PICS_IN_DIR (int): [see function description]
        """
        # shorten calls
        con = self.config
        # get the relevant min number
        MIN_PICS_IN_DIR = con["fps_reduce_fac"] * con["sequence_len"]
        return (MIN_PICS_IN_DIR)

    def get_datapack_folderpaths(self, data_packname):
        """return the paths of all folders for the given traindata-packname

        Args:
            data_packname (str): name of the desired data pack

        Returns:
            folderpaths (list): list of PosixPaths to the folder in the datapack
        """
        # get content per dir
        pack_main_dir = Path(
            f"{self.abs_path.parent}/" \
            f"{mt_hlp.name_constants('TRAIN_DATA_PACKS_DIR_NAME')}/" \
            f"{data_packname}/"
        )
        folderpaths = mt_hlp.get_folderpaths_in_dir(pack_main_dir)
        return (folderpaths)

    def identify_unvalid_folder(self, folderpaths):
        """check that all datafolders contain enough hand representations and
        sort those out, that do not have enough

        Args:
            folderpaths (list): the folderpaths to the videos in the datapack

        Returns:
            cleanded_folderpaths (list): sorted out the folderpath that do not
                                         have enough datapoint
        """
        cleanded_folderpaths = []
        # itterate over all videos in the given pack
        for videopath in folderpaths:
            num_samples = len(mt_hlp.get_filepaths_in_dir(videopath))
            # keep those that contain enough datapoint
            if num_samples >= self.MIN_PICS_IN_DIR:
                cleanded_folderpaths.append(videopath)
            # print a warning for those that are skipped
            else:
                print(f"Warning!\n" \
                      f"videodata from '{videopath.name}' not used, as it" \
                      f"does not contain enough frames" \
                      f"[self.MIN_PICS_IN_DIR = {self.MIN_PICS_IN_DIR}\n")
        return (cleanded_folderpaths)
    
    def redecue_if_clip(self, shuffled_stacked_filpaths, mode):
        """if clipping by max-data-volume is activated (due to restrict the
        maximum training times or to reduce skewness in the data due to
        different data preperations leading to different number of samples),
        clip the instances here to a comparable volume

        Args:
            shuffled_stacked_filpaths (list): the stacked filepaths. It's a list
                                              containing lists with paths per
                                              instance
            mode (str): in ["train", "val", "analyse"], determines which data
                        is treated; it's needed here as the clipping rules are
                        adapted to the mode
        
        Return:
            shuffled_stacked_clipped_filpaths (list): the same as the input, but
                                                      possibly reduced (clipped)
                                                      to a fix number
                                                      representing the volume
        """
        # just assign the original stacks and return them, if nothing is
        # getting clipped
        shuffled_stacked_clipped_filpaths = shuffled_stacked_filpaths
        # only enter the routine if clipping is activated
        if self.clip_max:
            # calculate the current volume of the dataset by multiplying the
            # number of instances with the sequence length
            curr_volume = (
                np.shape(shuffled_stacked_filpaths)[0] *
                np.shape(shuffled_stacked_filpaths)[1]
            )
            # initialise clipping values with false. if clipping is triggered,
            # they will be assigned in the conditions
            trigger_clipping = False
            # this variable will determine how may instances are to be kept to
            # reach the desired volume (if the dataset currently has a bigger
            # volume)
            take_n = False
            if mode == "train":
                if curr_volume > self.clip_data_vol_thresh:
                    take_n = int(
                        (self.clip_data_vol_thresh / curr_volume) *
                        np.shape(shuffled_stacked_filpaths)[0]
                    )
                    trigger_clipping = True
            elif mode == "val":
                if curr_volume > (self.clip_data_vol_thresh *
                    self.clip_train_val_ratio):
                    take_n = int(
                        ((self.clip_data_vol_thresh * 
                          self.clip_train_val_ratio) / curr_volume) *
                        np.shape(shuffled_stacked_filpaths)[0]
                    )
                    trigger_clipping = True
            # dont clip an analysation set
            elif mode == "analyse":
                pass
            # if clipping got triggered, clip the shuffled paths
            if trigger_clipping == True:
                shuffled_stacked_clipped_filpaths = (
                    shuffled_stacked_filpaths[:take_n]
                )
        return(shuffled_stacked_clipped_filpaths)

    def stack_paths(self, filepaths):
        """stack the paths in a way together as it is requested through the
        setting of the sequence_len and fps_reduce_fac, configured in the
        config file for the current run

        Args:
            filepaths (list): sorted possix paths, as list(list()) for every
                              video

        Returns:
            stacked_filpaths (list): stacked filepaths, not anymore per video,
                                     but per instance. every first level entry
                                     represents now one training instance (the
                                     paths to the files of it)
        """
        # shorten calls
        con = self.config

        stacked_filepaths = []
        # itterate through all videos in the training pack
        for video_paths in filepaths:
            # perform the desired fps reduction
            reduced_paths = self.reduce_paths(video_paths)
            # determine the number of frame-representations left
            frames_available = len(reduced_paths)
            for idx_0 in range(frames_available - con["sequence_len"] + 1):
                # get indexes and stacks
                idx_1 = idx_0 + con["sequence_len"]
                instance_paths = reduced_paths[idx_0:idx_1]
                # convert the PosixPaths to string, sothat tensorflow can
                # work with them
                instance_paths = [str(path) for path in instance_paths]
                # stack them all in the same list, as there is no need anymore
                # to seperate representations by video after this routine
                stacked_filepaths.append(instance_paths)

        return (stacked_filepaths)

    def reduce_paths(self, video_paths):
        """reduce the available frame representations by the desired factor.
        E.g. if the training data got created @24fps but the network shall
        work with an input of 6fps, one would set the "fps_reduce_fac" in
        the config dict to 4 and this routine will only keep every fourth
        path, so the input is just as if it was taken at 6fps

        Args:
            video_paths (list): the sorted paths to the data of an input video

        Returns:
            reduced_paths (list): if configured, the input list gets reduced to
                                  desired framerate
        """
        reduced_paths = video_paths[::self.config["fps_reduce_fac"]]
        return (reduced_paths)

    def load_vector_representations(self, instance):
        """load the paths assigned to the instance, return a matrix that
        is the desired sequential instance, in the final form to be fed to
        a network

        Args:
            instance (tf.Tensor): list of paths as byte strings that lead to
                                  csv files

        Returns:
            instance (tf.Tensor): tensor representation of the sequential input
            label (tf.Tensor): the motion class for the most recent coordinates
        """

        def _load_stacked_csvs(instance):
            """py-function to wrap for graph execution. enables easy loading of
            the stacked csv paths

            Args:
                instance (tf.Tensor): list of paths as byte strings that lead to
                                    csv files

            Returns:
                instance (tf.Tensor): tensor representation of the sequential
                                      input
                label (tf.Tensor): the motion class for the most recent
                                   coordinates
            """
            # a buffer list that will contain the hand representations for
            # as many steps back as desired
            buffer = []
            # initialise the potential label with datatype
            pot_label = np.float16(np.nan)

            # itterate through paths
            # NOTE: possibly parallelize this to use more cores
            for path in instance:
                # convert byte tensor to string
                this_path = path.numpy().decode("utf-8")
                # decode the byte tensor to string, load and transpose the data
                df = pd.read_csv(this_path, header=None).transpose()
                # adjust the loaded data to have columns and indexes all set up
                # right
                df.reset_index(drop=True, inplace=True)
                df.columns = df.iloc[0]
                df.drop(index=0, inplace=True)

                # NOTE: das hier gegen einen preprocessing layer mit strategie
                #		tauschen! Nur fürs erste developement hier mit drin...
                # df = df.fillna(1)

                # collect the lines from the files, appending the arrays
                # should be the fastest
                buffer.append(df.values[0])
                # grab potential label. the one from the last itteration will
                # be the one to use
                pot_label = np.float32(
                    this_path.split("/")[-1].split(".")[0].split("_")[1]
                )
            # convert the list of numpy arrays to a stacked array
            buffer = np.vstack(buffer)

            # assign the instance to a tensor with the right dataformat
            x = tf.convert_to_tensor(buffer, dtype=tf.float32)
            # one-hot encode the label to use losses like categorical
            # crossentropy
            y = tf.one_hot(np.array(pot_label), self.config["motion_classes"])

            return (x, y)

        # use py_function for the IO operations
        x, y = tf.py_function(
            func=_load_stacked_csvs,
            inp=[instance],
            Tout=([tf.float32, tf.float32])
        )

        # set the shapes, as they apparently do not get recognised automaticly
        # when using the py_function. If these lines are missing, wild things
        # happen
        # NOTE: das könnte man auch direkt auf x und y referenzieren,
        #		andererseits ist ein benutzer so gezwungen etwas mehr zu
        #		verstehen was hier passiert. das widerum könnte verhindern dass
        #		unvorhergesehenes passiert, darum ist es so gecoded
        x.set_shape([
            self.config["sequence_len"],
            self.config["initial_dimensions"]
        ])
        y.set_shape([
            self.config["motion_classes"]
        ])

        return (x, y)


def h5_to_tflight(h5_model):
    """convert a h5 tensorflow model to a tflight model

    Args:
        h5_model (tf.model): a trained model that shall be converted

    Returns:
        tflight_model (tf.model): the converted model
    """
    pass


def custom_layer_assignment():
    """return a collection of all custom layers with their names. This
    collection can be used to load models that rely on custom layers, without
    having to hardcode everywhere the names of the layers.
    This function should be redundant when using the decorator
    @tf.keras.utils.register_keras_serializable() at custom layers, but
    because of a tf-bug, this doesnt seem to work for the current version
    and thus this function shall be a quiet clean workaround until the
    serialization decorator works as expected

    NOTE: IT WORKS NOW WITHOUT THE STUFF HERE

    Returns:
        cus_lay_assignment (dict): the assignment of the custom layers with
                                   their names
    """
    cus_lay_assignment = {
        "Hand_Input_Sorter": cus_lay.Hand_Input_Sorter,
        "Man_Dim_Red": cus_lay.Manual_Dim_Reducer,
        "Hand_Imputer": cus_lay.Hand_Imputer,
        "Hand_Normz": cus_lay.Normalizer,
        "NaNs_to_zero": cus_lay.NaN_to_zero
    }
    return (cus_lay_assignment)


def get_dev_ds():
    """provide a dataset that has dimensions and everything like the real
    one, but only a few instances with easy evaluatable values.
    This dataset can be used for developement of custom layers, functions
    and nets in general, to test theire behaviour.

    Returns:
        Returns:
            ds (tf.Dataset): trainable/predictable dataset
    """
    # tune parameters as desired. Note that COO_DIMS can not be changed due to
    # keep easy values and easy programming
    seq_len = 5
    COO_DIMS = 3  # not dynamically implemented
    instances_n = 2
    batch_size = 2

    # initialise a dummy label that will added to every instance
    dummy_label = tf.convert_to_tensor(np.array([1, 0, 0, 0]))

    # outer storage to be transformed to the dataset
    gen = []
    # first loop determines how many instances will be included
    for i in range(instances_n):
        # temp instance storage
        inst = []
        # this loop determines how many sequences are stacked
        for i in range(seq_len):
            # initialise a tensor with easy evaluatable values
            X = tf.convert_to_tensor(np.array([.1 * i, 0.6, 1]))
            inst.append(X)
        # store the instance
        gen.append(inst)
    # create a tf-dataset from the saved tensors
    ds = tf.data.Dataset.from_tensor_slices(gen)
    # add dummy labels
    ds = ds.map(lambda X: (X, dummy_label))
    # batch and prefetch them like for the real dataset
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return (ds)

def get_predictions_from_ds(model, dataset, verbose=False):
    """helper function to return Y_true and Y_pred for a given model and
    dataset

    Args:
        model (tf.model): the model to evaluate
        dataset (tf.dataset): the dataset
        verbose (bool): if stuff shall be printed or not
    
    Returns:
        Y_true (np.array): the labels for the whole dataset
        Y_pred (np.array): the predictions for the whole dataset
    """
    if verbose:
        print("\nExtract labels and predictons from dataset")
    # create storage and pointer variables
    i = 0
    Y_true = []
    Y_pred = []
    dataset_batches = tf.data.experimental.cardinality(dataset).numpy()
    silence = False if verbose else True
    # loop over the dataset, extract x and y
    for x, y in tqdm(dataset, total=dataset_batches, disable=silence):
        #if verbose:
        #    print(f"\rcurrently at batch: {i} / {dataset_batches}", end="")
        # count up and print as information. i will equal the number of batches
        #i += 1
        # get y
        Y_true.append(y.numpy())
        # predict the batch with the given model, save the prediction
        Y_pred.append(model.predict(x))
    # keep the class with the max propability as label/pred. that also means,
    # here the probability of each prediction is dropped
    Y_true = np.argmax(np.concatenate(Y_true), axis=1)
    Y_pred = np.argmax(np.concatenate(Y_pred), axis=1)
    # print all labels and predictions
    if verbose:
        print("\nY_true:\n", Y_true)
        print("Y_pred:\n", Y_pred)
        print("Y_true shape:\n", Y_true.shape)
        print("Y_pred shape:\n", Y_pred.shape)
    return(Y_true, Y_pred)



if __name__ == "__main__":
    pass
