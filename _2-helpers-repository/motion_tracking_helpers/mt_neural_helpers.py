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
from pympler import asizeof
import sklearn as sk
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
import sys
import tensorflow as tf
from tqdm import tqdm
import wandb
from wandb.keras import WandbCallback

# modules from this package
from . import custom_layers as cus_lay
from . import motion_tracking_helpers as mt_hlp
from . import mt_plot_helpers as plt_hlp
from ._olds import Old_DataBuilder_class


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


class ModelCheckpointCombineModelsCallback(tf.keras.callbacks.Callback):
    """custom callback, to save a combined-best-model at a given checkpoint.
    This is needed if a dataset was preprocessed for training to speed up the
    training, but a fully-functional model shall be saved, that can operate on
    unprepared input data
    """
    def __init__(self, prep_model, input_shape, model_save_path):
        """specify everything needed to save a combined model when a a new best
        validation loss is hitted

        Args:
            prep_model (tf.model): the model used at preprocessing
            input_shape (tuple): contains the input shape, which is needed to
                                 compile the combined model
            model_save_path (str): the path to save the model at
        """
        # assign the inputs
        self.prep_model = prep_model
        self.input_shape = input_shape
        self.model_save_path = model_save_path
        # monitor against the validation loss, create a variable for that
        self.best_val_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        """gets called after every epoch. Check if a new best validation loss
        was reached and possibly save a model, which is the combination of the
        currently trained model and the model used to prepare the data
        """
        # if the epoch is a new best, save the model
        if logs["val_loss"] < self.best_val_loss:
            # save the current val_loss
            self.best_val_loss = logs["val_loss"]
            # create a combined model
            full_model = tf.keras.models.Sequential()
            # deterine the input shape
            full_model.add(
                tf.keras.layers.InputLayer(
                    input_shape=(self.input_shape)
                    )
            )
            # itterate over the prepocessing model's layers
            for layer in self.prep_model.layers:
                full_model.add(layer)
            # itterate through the current layers (the model is available as
            # the class inherits from tf.[...].Callback)
            for layer in self.model.layers:
                full_model.add(layer)
            # compile the model
            full_model.compile()
            # save the model
            full_model.save(self.model_save_path, overwrite=True)


def callback_initialiser(config, abs_path, run, add_prep_model=False,
        prep_model=False):
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
        add_prep_model (bool): if a preprocessing model shall be added to the
                               saved model at checkpoints
        prep_model (False || tf.model): if add_prep_model==True, the model to
                                        put in front of the saved model

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
        
        # create the model's saving path, that needs to be a str for tf-Callback
        model_save_path = str(model_save_path) + "/best_model.h5"

        # add the best-model-callback. wether the standard tf one or a custom
        # save callback, if a preprocessing model shall be stacked in front of
        # the currently trained one
        if not add_prep_model:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_save_path,
                    monitor=config["monitor_metric"],
                    verbose=0,
                    save_best_only=True
                )
            )
        else:
            # gather the input shape for the combined model
            input_shape = (config["sequence_len"], config["initial_dimensions"])
            # initialise the custom callback and append it
            CustomModelChekpoint = ModelCheckpointCombineModelsCallback(
                prep_model, input_shape, model_save_path
            )
            callbacks.append(CustomModelChekpoint)

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


class DataBuilder(Old_DataBuilder_class.DataBuilder):
    """keep the old builder available here for easy comparison and to keep
    old scripts functional
    """
    def __init__(self, config, arch_path, is_train=True, clip_max=False, **kw):
        """initialise the deprecated DataBuilder
        """
        print("\n\nDEPRECATION NOTE:\nA way faster DataBuilder is available" \
              " under the class name 'DataBuilder_v2'. It works as drop in" \
              " replacement\n\n")
        # initialise the needed init's
        super(DataBuilder, self).__init__(config, arch_path)


class DataBuilder_v2():
    def __init__(self, config, arch_path, is_train=True, clip_max=False,
            shuffle=True):
        """set initial vars, that are nice to grab over and
        over again and that are not to specific

        Args:
            config (dict): a dictionary, that contains all needed variables to
                           run the model and document it
            arch_path (PosixPath): path of the triggering script (-> from the 
                                   99_Architectures-dir). This is needed to
                                   access the data stored locally
            is_train (bool): if we are in training mode or not
            clip_max (bool): if instances shall be reduces (-> clipped) when
                             their volume (num_n*seq_len) overcome a certain
                             threshold
            shuffle (bool): if the instances shall be shuffeled
        """
        # assign everything needed
        self.config = config
        self.arch_path = arch_path
        self.is_train = is_train
        self.shuffle = shuffle
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
        # these values are used in the tf.slice method, every time a batch gets
        # requested
        self.start_at_0 = tf.constant([0])
        self.seq_len = tf.constant([self.config["sequence_len"],-1])

    def build(self, mode, verbose=True):
        """paths to files to scan

        Args:
            mode (str): in ["train", "val", "analyse"], determines which data
                        will be the base for the dataset that is returned
            verbose (bool): if stuff shall be printed or not

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
        print()
        logging.info(f"building the dataset for: '{data_packname}'")

        logging.debug("collecting patchs, sort out unvalid")
        folderpaths = self.get_datapack_folderpaths(data_packname)
        # check if all folderpath contain enough datapoints for the stacking
        folderpaths = self.identify_unvalid_folder(folderpaths)
        
        logging.debug("get filepaths from every video")
        # get the filepaths for every video
        filepaths = [mt_hlp.get_filepaths_in_dir(folderpath) for folderpath in
                     folderpaths]
        
        logging.debug("sort the paths")
        # sort paths ascending by the assigned framenumber to ensure that the
        # data is consistent
        filepaths = [sorted(paths, key=lambda x: int(x.name.split("_")[0])) for
                     paths in filepaths]
        
        logging.debug("stack the paths")
        # stack the paths according to the desired sequential input shape and
        # the desired fps reduction. Also convert the PosixPaths to strings on
        # the fly
        stacked_filpaths = self.stack_paths(filepaths)
        
        logging.debug("flatten the paths, create set with unique ones")
        # examine which paths uniquly exist in the requested data
        unique_paths = (
            set([item for sublist in stacked_filpaths for item in sublist])
        )
        
        logging.debug("sort the unique paths in the same order as the stacked" \
                      " paths to ensure correct lookup behaviour")
        """
        unique_paths = sorted(
            unique_paths_pre, key=lambda x: int(
                (x.split("/")[-1]).split("_")[0]
                )
        )
        """

        unique_paths_sorted = []
        # sort the paths - whoses sorting got lost by the set function - the
        # same way than the stacked_paths are sorted
        for sample in folderpaths:
            # extract the current sample name
            sample_name = str(sample.name).split("/")[-1]
            # get all unique paths from the given sample
            sample_paths = filter(
                lambda x: x.split("/")[-2] == sample_name, unique_paths
            )
            # sort them according to the corresponding frame numbers
            sample_paths_sorted = sorted(
                sample_paths, key=lambda x: int(
                    (x.split("/")[-1]).split("_")[0]
                    )
            )
            # save them
            unique_paths_sorted.extend(sample_paths_sorted)
        
        logging.debug("load representations and labels, create lookups")
        # create the lookup lists with the instances and theire labels, by
        # loading every unique path and extracting the information, as well
        # as a dictionary that saved each path with an explicit number. This
        # will be used to replace the stacked_filepaths with stacked lookup
        # numbers
        inst_lookup_list = []
        label_lookup_list = []
        lookup_dict = {}
        # here, an initial big load is performed
        if verbose:
            print(f"Create major lookup table for ds: '{data_packname}'")
        silence = False if verbose else True
        for i, filename in tqdm(
                enumerate(unique_paths_sorted),
                total=len(unique_paths_sorted),
                disable=silence):
            hand_representation = pd.read_csv(filename, header=None)[1].values
            inst_lookup_list.append(hand_representation)
            try:
                label_lookup_list.append(
                    int(filename.split("_")[-1].split(".")[0])
                )
            except:
                print(filename)
            lookup_dict[filename] = i
        
        logging.debug("convert lookups to tensors")
        # convert the lookup lists to tensors for efficient use with tf.data
        self.lookup_inst_tensor = tf.constant(np.array(inst_lookup_list))
        self.lookup_label_tensor = tf.constant(np.array(label_lookup_list))
        
        logging.debug("get end-idx for lookup per instance, stack them")
        # get from the stacked_filepaths, whereof each represents an instance,
        # the last path, look up the corresponding lookup-number, and create
        # the new instance-representations (-> end_idx_s) from those
        end_idx_s = []
        for instance in stacked_filpaths:
            idx = lookup_dict[instance[-1]]
            end_idx_s.append(idx)
        
        logging.debug("shuffle the idx's")
        # shuffle the stacked idx_s at this point (if shuffeling is activated),
        # so that no shuffling needs to be performed during prefetching, which
        # would be quiet a pain with regard to the low number of input videos
        # that are especially long
        if self.shuffle:
            shuffled_stacked_idx_s = sk.utils.shuffle(
                end_idx_s,
                random_state=self.config["rnd_state"])
        else:
            # dummy assignment
            shuffled_stacked_idx_s = end_idx_s
        
        logging.debug("possibly clip them")
        # if clipping is activated, possibly take an appropriate number n of
        # the first elements to reduce the dataset to a maximum fixed volume. As
        # the shuffeling of the instances happend before, this easy approach is
        # very efficient and valid
        shuffled_stacked_clipped_idx_s = self.redecue_if_clip(
            shuffled_stacked_idx_s, mode
        )
        
        logging.debug("convert to tensor")
        # add a dimension for easy handeling later on
        prep_tensor = tf.expand_dims(
            tf.constant(shuffled_stacked_clipped_idx_s), axis=1
        )
        
        # free up space by deleting not needed variables
        logging.debug("free up space by deleting not needed variables")
        del folderpaths, filepaths, stacked_filpaths, unique_paths
        del unique_paths_sorted, inst_lookup_list, label_lookup_list,
        del lookup_dict, end_idx_s, shuffled_stacked_idx_s
        del shuffled_stacked_clipped_idx_s
        # this is the main RAM occupying object left. Log it's size
        size_inst_lookup = get_size_of_tensor_MB(self.lookup_inst_tensor)
        logging.info(f"size of lookup tensor: {size_inst_lookup:.2f} MB")
        logging.debug("pass to tf.data: map lookup, batch, prefetch")
        
        # create the dataset from the shuffled stacked lookup numbers
        ds = tf.data.Dataset.from_tensor_slices(
            prep_tensor
        )
        
        # load the hand representations based on the lookup number and stack
        # them
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
            f"{self.arch_path.parent}/" \
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
    
    def redecue_if_clip(self, shuffled_stacked_idx_s, mode):
        """if clipping by max-data-volume is activated (due to restrict the
        maximum training times or to reduce skewness in the data due to
        different data preperations leading to different number of samples),
        clip the instances here to a comparable volume

        Args:
            shuffled_stacked_idx_s (list): the stacked lookup-end_idxs, thus
                                           the instances themselve
            mode (str): in ["train", "val", "analyse"], determines which data
                        is treated; it's needed here as the clipping rules are
                        adapted to the mode
        
        Return:
            shuffled_stacked_clipped_idx_s (list): the same as the input, but
                                                   possibly reduced (clipped) to
                                                   a fix number representing the
                                                   volume
        """
        # just assign the original stacks and return them, if nothing is
        # getting clipped
        shuffled_stacked_clipped_idx_s = shuffled_stacked_idx_s
        # only enter the routine if clipping is activated
        if self.clip_max:
            # calculate the current volume of the dataset by multiplying the
            # number of instances with the sequence length
            curr_volume = (
                np.shape(shuffled_stacked_idx_s)[0] *
                self.config["sequence_len"]
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
                        np.shape(shuffled_stacked_idx_s)[0]
                    )
                    trigger_clipping = True
            elif mode == "val":
                if curr_volume > (self.clip_data_vol_thresh *
                    self.clip_train_val_ratio):
                    take_n = int(
                        ((self.clip_data_vol_thresh * 
                          self.clip_train_val_ratio) / curr_volume) *
                        np.shape(shuffled_stacked_idx_s)[0]
                    )
                    trigger_clipping = True
            # dont clip an analysation set
            elif mode == "analyse":
                pass
            # if clipping got triggered, clip the shuffled paths
            if trigger_clipping == True:
                shuffled_stacked_clipped_idx_s = (
                    shuffled_stacked_idx_s[:take_n]
                )
        return(shuffled_stacked_clipped_idx_s)

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
        """lookup the instances, that are meant to be used with the given
        instance end_idx-lookupcode (that is what comes flying into this
        function). Return a matrix thatn is the desired sequential instance, in
        the final form to be fed to a network

        Args:
            instance (tf.Tensor): a single number, that represents the end-index
                                  of the hand representations to be loaded with
                                  the choosen sequence_len

        Returns:
            instance (tf.Tensor): tensor representation of the sequential input
            label (tf.Tensor): the motion class for the most recent coordinates
        """
        # get the slicing start point from the lookup tensor by subtracting the
        # given sequence-length from the instances-start point. This is also
        # valid for fps-reduced-data, as in the building process, these paths
        # were let out, which means the lookuptable has consecutive numbers
        # in every case and can be looked up here with this easy approach
        start_point = tf.math.subtract(instance+1, self.seq_len[0])

        # the slicing starts in the instance axis at "start_point" and for
        # each representation at "self.start_at_0" which equals 0
        slice_start = tf.concat((start_point, self.start_at_0), axis=0)
        
        # slice out the instance from the lookup table
        x =  tf.cast(
            tf.slice(
                self.lookup_inst_tensor,
                slice_start,
                self.seq_len
            ),
            tf.float32
        )
        
        # ...and also lookup the label
        y_pre = self.lookup_label_tensor[instance[0]]
        y = tf.one_hot(y_pre, self.config["motion_classes"])

        # set the shapes, as they apparently do not get recognised automaticly
        # when using the py_function. If these lines are missing, wild things
        # happen
        # NOTE: mit dem neuen approach könnte man mal versuchen das weg zu
        #       lassen
        """
        x.set_shape([
            self.config["sequence_len"],
            self.config["initial_dimensions"]
        ])
        y.set_shape([
            self.config["motion_classes"]
        ])
        """

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
    """helper function to return X, Y_true and Y_pred for a given model and
    dataset

    Args:
        model (tf.model): the model to evaluate
        dataset (tf.dataset): the dataset
        verbose (bool): if stuff shall be printed or not
    
    Returns:
        X (np.array): the underlaying x data
        Y_true (np.array): the labels for the whole dataset
        Y_pred (np.array): the predictions for the whole dataset
    """
    # create storage vars
    X = []
    Y_true = []
    Y_pred = []
    dataset_batches = tf.data.experimental.cardinality(dataset).numpy()
    # loop over the dataset, extract x and y
    # NOTE: Predicting the data after the loop at once is approx. 10 % faster.
    #       This approach is kept anyway, as the progress bar is valued higher
    #       than the speed improvements on this time-uncritical-part. This might
    #        need to be re assesed at future uses...
    print("\nExtract labels and predictons from dataset")
    for x, y in tqdm(dataset, total=dataset_batches):
        # save x & y
        X.append(x.numpy())
        Y_true.append(y.numpy())
        # predict the batch with the given model, save the prediction
        Y_pred.append(model.predict(x))
    # concetenate the collected data to proper numpy arrays
    X = np.concatenate(X)
    Y_true = np.concatenate(Y_true)
    Y_pred = np.concatenate(Y_pred)
    # print all labels and predictions; once the original for clearity and
    # once shrinked to the label with max probability
    if verbose:
        print("\nX:\n", X)
        print("X shape: ", X.shape)
        print("\n---------")
        print("\nY_true:\n", Y_true)
        print("Y_true shape: ", Y_true.shape)
        print("\nY_pred:\n", Y_pred)
        print("Y_pred shape: ", Y_pred.shape)
        print("\n---------")
        print("\nY_true-max_prob:\n", np.argmax(Y_true, axis=1))
        print("Y_true-max_prob shape: ", np.argmax(Y_true, axis=1).shape)
        print("\nY_pred-max_prob:\n", np.argmax(Y_pred, axis=1))
        print("Y_pred-max_prob shape: ", np.argmax(Y_pred, axis=1).shape)
        print("\n")

    return(X, Y_true, Y_pred)


def get_size_of_py_obj_MB(obj):
    """Return the size of a python object in megabyte. By using pympler's 
    asizeof, nested structures are also concidered

    Args:
        obj (-): any python object. also numpy objects shall work
    
    Returns:
        size (float): the size in MB
    """
    # calculate the size in megabytes
    size_mb = asizeof.asizeof(obj) / (1024*1024)
    return(size_mb)


def get_size_of_tensor_MB(tensor):
    """dynamically calculate the RAM needed by a tensor

    Args:
        tensor (tf.Tensor): tensor of any shape
    
    Returns:
        size (float): the size in MB
    """
    # the dtype needs to be determined to know how many bytes are used per entry
    if tensor.dtype == tf.float16:
        bytes_per_entry = 2
    elif tensor.dtype == tf.int16:
        bytes_per_entry = 2
    elif tensor.dtype == tf.float32:
        bytes_per_entry = 4
    elif tensor.dtype == tf.int32:
        bytes_per_entry = 4
    elif tensor.dtype == tf.float64:
        bytes_per_entry = 8
    elif tensor.dtype == tf.int64:
        bytes_per_entry = 8
    else:
        logging.warning("Datatype for byte analysation of tensor not found." \
              " Returning size of 0 MB")
        bytes_per_entry = 0
    # itterate over the dimensions
    init_mul = 1
    for num in tensor.get_shape():
        init_mul *= num
    # include the dtype-byte size
    size_in_bytes = init_mul * bytes_per_entry
    # conver to megabytes
    size_mb = size_in_bytes / (1024*1024)
    return(size_mb)


def save_analysation_data(Y_max_prob, pass_args, Eval_Specs):
    """trigger this funcion to save analysation insights for model-datapack
    combinations to an Eval_Specs-object, to possibly later compare metrics
    between different models and data

    Args:
        Y_max_prob (tuple(np.array)): contains the true- and pred- labels
        pass_args (tuple): contains 3 arguments that describe how the given
                           predictions were derived:
                                run_name (str): the name of the run, in which
                                                the model has it's origin
                                datapack_name (str): the name of the datapack
                                                     that's currently analysed
                                config (dict): config dict from the run
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder and the current run
    """
    # unpack the bundled inputs
    Y_true, Y_pred = Y_max_prob
    run_name, datapack_name, config = pass_args

    # extract major metrics and stuff
    accuracy = accuracy_score(Y_true, Y_pred)
    conf_mat = confusion_matrix(Y_true, Y_pred)
    classes = plt_hlp.get_classes_from_Y((Y_true, Y_pred))
    class_rep_dict = classification_report(
        Y_true, Y_pred, target_names=[str(i) for i in list(classes)],
        zero_division=0, output_dict=True
    )
    class_rep_print = classification_report(
        Y_true, Y_pred, target_names=[str(i) for i in list(classes)],
        zero_division=0
    )
    metr_macro = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0
    )
    metr_weighted = precision_recall_fscore_support(
        Y_true, Y_pred, average="weighted", zero_division=0
    )
    support = Y_true.shape[0]

    # save them
    Eval_Specs.add_information(
        run_name,
        datapack_name,
        accuracy=accuracy,
        conf_mat=conf_mat,
        classes_in_data=classes,
        classification_report_dict=class_rep_dict,
        classification_report_print=class_rep_print,
        precision_recall_fscore_support_macro=metr_macro,
        precision_recall_fscore_support_weighted=metr_weighted,
        support=support
    )


def fetch_wandb_project_data(project, entity="beejayk", timeout=20):
    """fetch data from the wandb API to enable lookups and further works
    with it

    Args:
        project (str): the name of a project on wandb, of which the data
                       shall be fetched
        entity (str): the name of the entity behind the project
        timeout (int): the allowed API timeout
    
    Returns:
        runs_df (pd.DataFrame): a dataframe, containing all the information
    """
    # enter the wandb space
    api = wandb.Api(timeout=timeout)
    entity, project = entity, project
    runs = api.runs(entity + "/" + project)

    # create temp storrages
    summary_list, config_list, name_list, id_list = [], [], [], []
    print(f"fetching the run data for the given project: '{project}'")
    for run in tqdm(runs):
        # .summary contains the output keys/values for metrics like accuracy
        #  call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        # remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})
        # .name is the human-readable name of the run.
        name_list.append(run.name)
        # save the runs id for further api communication
        id_list.append(run.id)

    # summarize everything in a dataframe for easy continuous work
    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
        "id": id_list
        })
    
    return(runs_df)


def create_preprocessed_dataset(model, dataset):
    """transform a given dataset to a new, preprocessed one. Note, that the
    given dataset needs to be composed of train and label data.

    Args:
        model (tf.model): the model to evaluate
        dataset (tf.dataset): the dataset
    
    Returns:
        prep_dataset (tf.dataset): the preprocessed dataset
    """
    # transform the dataset
    _, Y_train_prep, X_train_prep = get_predictions_from_ds(model, dataset)
    # get the orginally used batch size
    batch_size = next(iter(dataset))[0].shape[0]
    # recreate it
    prep_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train_prep, Y_train_prep)
        )
    # batch it and make it buffer automaticly
    prep_dataset = prep_dataset.batch(batch_size)
    prep_dataset = prep_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    return(prep_dataset)



if __name__ == "__main__":
    pass
