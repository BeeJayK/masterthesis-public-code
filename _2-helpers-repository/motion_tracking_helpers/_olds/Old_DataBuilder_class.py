#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:06:20 2021

@author: marlinberger

Save the old DataBuilder. The new one is MUUUUUUUCH faster, and consumes approx.
the same amount of RAM. Different tecniques are used to achieve that, the
ideas are noted in this builder already, at the py_func, as there was the
bottleneck
"""
# python packages
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn as sk
import tensorflow as tf

# modules from this package
from .. import motion_tracking_helpers as mt_hlp


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
            # NOTE: quick speed improvements may be achieved by only loading
            #       the csv in the loop and try to make all the df-ops
            #       vectorized at one
            # NOTE: real improvements (possibly lazer-blazer-fast) could be
            #       achieved by reworking the function into pure tf-code like
            #       this: load all hand representations in the init-function,
            #       transform them in the same fashion as done here and convert
            #       them into two large tensors, that contain all hand
            #       representations sorted.
            #       After that, create the instances just as it's done here,
            #       but instead of bundle paths to instances, bundle indexes
            #       to instances. These indexes refer to the previous created
            #       large tensor(s) for x & y.
            #       In here, every instance just performs a simply lookup,
            #       loading the instances and theire labe from the init's
            #       tensors.
            #       BOOM! No py_func needed, much more efficient loading,
            #       possibly large speed improvements.
            #       Further RAM-saving improvements, by not stacking all the
            #       numbers for the instances together, but only the start-index
            #       as single number. As the seq_len is given in the init with
            #       the config dict, the slice to extract could just be built
            #       dynamically, what should not be a bottelneck (test speed
            #       difference!) but saves a tremandous amount of RAM.
            #       EDIT: With tf.slice this can even be obtained directly!
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