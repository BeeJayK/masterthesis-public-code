#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:57:13 2021

@author: marlinberger

Save assignements of the names of the columns from the data and theire
positions. This way, tf-data input pipelines etc. are able to be programmed
efficiently and reliable.
"""
# python packages
import json
import numpy as np
from pathlib import Path


class CSV_Architecture_v2():
    """the assignments of the variables by name and position, version 2
    """

    def __init__(self):
        # set up general assignment
        self.assignment = self.assignment()
        # get indexes to the handedness prediction and its confidence
        self.handedness_0_idx, self.handedness_1_idx = self.handedness_pred()
        self.handedness_0_prob_idx, self.handedness_1_prob_idx = (
            self.handedness_pred_prob()
        )
        # get index-arrays for the coordinates and handedness results for each
        # hand
        self.hand_0_idxs, self.hand_1_idxs = self.hand_idxs()
        # get index-arrays for both hands, that only contain the indexes of the
        # coordinate points
        self.hand_0_idxs_coo_only, self.hand_1_idxs_coo_only = (
            self.hand_0_idxs[2:],
            self.hand_1_idxs[2:]
        )
        # create more subarrays, with indexes for the coordinates splitted
        # by dimension
        self.hand_0_x, self.hand_0_y, self.hand_0_z = self.split_by_dimension(
            self.hand_0_idxs
        )
        self.hand_1_x, self.hand_1_y, self.hand_1_z = self.split_by_dimension(
            self.hand_1_idxs
        )
        # ~intro-value-idx's: timestamp1, timestamp2, no_hands_detected_flag.
        # these indexes will be droped by the man_dim_reduce-layer
        self.vector_intro_idxs = np.array([0, 1, 2])
        # indexes of additional information (meaning no coordinates), that might
        # contain usefull informations. Depending on how many values are in
        # here, it might make sense to pass these through skip connections.
        # For this architecture, these informations are espacially the
        # handedness informations
        self.relevant_add_infos_idxs = np.array([
            self.handedness_0_idx,
            self.handedness_0_prob_idx,
            self.handedness_1_idx,
            self.handedness_1_prob_idx
        ])
        # gather the indexes of some points that might representate already
        # enough information of the hand
        self.sig_5_points_0_idxs, self.sig_5_points_1_idxs = (
            self.gather_5_sig_points()
        )

        # this is the value at idx 3&68, if no hand was detected
        self.no_hand_detected_val = -1

    def assignment(self):
        """heart of the class. load the representation of the structure. If
        desired, look up the json file to see which line is assigned to which
        variable

        Returns:
            assignment_dict (dict): the assignment, as dictionary composed
                                    of strings
        """
        # set up the path to the assognment json
        path = (
            Path(__file__).parent / \
            "coordinate_assignments" / \
            "csv_assignment_v2.json"
        )
        # open, load and assign the file
        with open(path) as json_file:
            assignment_dict = json.load(json_file)
        return (assignment_dict)

    def handedness_pred(self):
        """return the indexes to the mark of the handedness result, meaning
        the index for hand_0 and hand_1, where the information is carried, if
        it is a right or left hand

        Returns:
            handedness_0_idx (int): idx for handedness result for hand_0
            handedness_1_idx (int): idx for handedness result for hand_1
        """
        handedness_0_idx = 3
        handedness_1_idx = 68
        return (handedness_0_idx, handedness_1_idx)

    def handedness_pred_prob(self):
        """return the indexes to the probability of the handedness result,
        meaning the index for hand_0 and hand_1, where the information is
        carried, how confidence mediapipe is about the right-or-left prediction

        Returns:
            handedness_0_prob_idx (int): idx for handedness probsability for
                                         hand_0
            handedness_1_prob_idx (int): idx for handedness probsability for
                                         hand_1
        """
        handedness_0_prob_idx = 4
        handedness_1_prob_idx = 69
        return (handedness_0_prob_idx, handedness_1_prob_idx)

    def hand_idxs(self):
        """compute the idx's that belong to the seperate hands coordinates and
        handedness predictions.
        These arrays will be used to conditioanally swap the hands in custom
        layers etc.

        Returns:
            hand_0_idxs (np.array): idx's of hand 0 coordinates
            hand_1_idxs (np.array): idx's of hand 1 coordinates
        """
        hand_0_idxs = np.array(range(3, 68))
        hand_1_idxs = np.array(range(68, 133))
        return (hand_0_idxs, hand_1_idxs)

    def gather_5_sig_points(self):
        """gather the indexes of some points that might representate already
        enough information of the hand. These indexes are used in manual-
        reduction-layer to reduce the amount of data points.
        The points choosen are:
            0: the wrist coordinate
            9: a central hand coordinate
            3: the second upmost thumb point
            7: the second upmost index-finger point
            15: the second upmost ring-finger point

        Return the indexes of these 3D points for both hands seperate. The
        arrays will have the size 5(points)x3(dims)=15

        Returns:
            sig_5_points_0_idxs (np.array): idx's of hand 0 coordinates
            sig_5_points_1_idxs (np.array): idx's of hand 1 coordinates
        """
        # for hand_0
        sig_5_points_0_idxs = np.array([
            # point 0
            5, 6, 7,
            # point 9
            32, 33, 34,
            # point 3
            14, 15, 16,
            # point 7
            26, 27, 28,
            # point 15
            50, 51, 52
        ])
        # for hand_1
        sig_5_points_1_idxs = np.array([
            # point 0
            70, 71, 72,
            # point 9
            97, 98, 99,
            # point 3
            79, 80, 81,
            # point 7
            91, 92, 93,
            # point 15
            115, 116, 117
        ])
        return (sig_5_points_0_idxs, sig_5_points_1_idxs)

    def split_by_dimension(self, hand_coordinates_idxs):
        """return three seperate arrays, that represent all indexes for the
        different dimension of the coordinates of one hand.
        This can be used in custom layers to get rid of single dimensions and
        stuff

        Args:
            hand_coordinates_idxs (np.array): idx's assigned to one hand

        Returns:
            hand_x_idxs (np.array): indexes, belonging to the x dimension
            hand_y_idxs (np.array): indexes, belonging to the y dimension
            hand_z_idxs (np.array): indexes, belonging to the z dimension
        """
        # super important: exclude the handedness results that stand in front
        hand_coordinates_idxs = hand_coordinates_idxs[2:]
        # extract the coordinates
        xs = [idx for idx in hand_coordinates_idxs[0::3]]
        ys = [idx for idx in hand_coordinates_idxs[1::3]]
        zs = [idx for idx in hand_coordinates_idxs[2::3]]

        # convert to numpy arrays
        hand_x_idxs, hand_y_idxs, hand_z_idxs = (
            np.array(xs), np.array(ys), np.array(zs)
        )
        return (hand_x_idxs, hand_y_idxs, hand_z_idxs)

    def decode(self, mode, name=None, position=None):
        """return the position to a given name or vice versa

        Args:
            mode (str): in ["pos2name", "name2pos"]. determines what is given
            name (str): the column name thats position is to return
                        (mode=="name2pos")
            position (int): the position thats name is to return
                            (mode=="pos2name")

        Return:
            name (str) || pos (int): the desired encoding of the input
        """
        # return mode depended what is desired
        if mode == "pos2name":
            name = self.assignment[str(position)]
            return (str(name))
        if mode == "name2pos":
            position = list(
                self.assignment.keys())[
                list(self.assignment.values()).index(name)
            ]
            return (int(position))
