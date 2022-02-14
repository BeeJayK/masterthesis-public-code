#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:10:09 2021

@author: marlinberger

script contains custom layers for neural networks
"""

# python packages
import logging
import numpy as np
from tensorflow import keras
import tensorflow as tf

# modules from this package
from . import coordinate_position_assignment as cpa


def check_import():
    """use this function to check if the import worked
    """
    logging.debug("\nyou made it in\n")


# decorator ensures easy loading of the custom layers
@tf.keras.utils.register_keras_serializable(
    package="Custom", name="Hand_Input_Sorter")
class Hand_Input_Sorter(keras.layers.Layer):
    """First step in the input pipeline for data derived from inf1. Dimensions
    are to be kept original at this point, sothat further layers can refer
    to the columns deterined for the csv-structure.
    (see script: coordinate_position_allignment)
    Goal of this layer is to ensure, the right- and left- hand's coordinates
    are always at the same indexes, so this layer possibly swaps them.
    """

    def __init__(self, strategy="mp_max_prob", csv_struct="v2",
                 debug_mode=False, name="Hand_Input_Sorter", **kwargs):
        """initialise everything needed for this layer

        Args:
            strategy (str): in ["mp_max_prob"]
            csv_struct (str): the structure of the used data. in ["v2"]
            debug_mode (bool): activates some prints if enabled
            name (str): name of the layer. defaults to tha class name
        """
        # save strategy, structure and debug-state
        self.strategy = strategy
        self.csv_struct = csv_struct
        self.debug_mode = debug_mode

        # initialise file-structure based variables
        if self.csv_struct == "v2":
            # structure object for csv-file-based-structure version 2
            self.arch = cpa.CSV_Architecture_v2()
            # idxs for handedness results and probabilities
            self.pos_0_idx = self.arch.handedness_0_idx
            self.pos_1_idx = self.arch.handedness_1_idx
            self.prob_0_idx = self.arch.handedness_0_prob_idx
            self.prob_1_idx = self.arch.handedness_1_prob_idx
            # arrays with indexes assigned to one hand
            self.idxs_0 = self.arch.hand_0_idxs
            self.idxs_1 = self.arch.hand_1_idxs
            # extract swap pattern based on the object
            len_arch = len(self.arch.assignment)
            all_idxs = np.array([i for i in range(len_arch)])
            # swap indexes
            all_idxs[self.idxs_0], all_idxs[self.idxs_1] = (
                all_idxs[self.idxs_1], all_idxs[self.idxs_0]
            )
            # prepare for use with tf-scatter
            self.swap_pattern = tf.constant(np.expand_dims(all_idxs, -1))

        # initialise strategy based variables
        if strategy == "mp_max_prob":
            """sorting will put the left hand on the first idxs, the right
            to the last. It will also be ensured, that if only one hand
            is given, it is on the right position. And if two left or two right
            hands are detected, the one with the higher handedness probability
            will be handeld as true.
            """
            # tf variables to later coordinate conditional decisions
            # ------------
            # variable to store left-right information per instance
            self.hs = tf.Variable(
                [0, 0], dtype=tf.float32, name="handedness_pred"
            )
            # variable to store left-right probability per instance
            self.hs_prob = tf.Variable(
                [0, 0], dtype=tf.float32, name="handedness_pred_conf"
            )
            # extract nan informations
            self.nan_mask = tf.Variable(
                [0, 0], dtype=tf.float32, name="nan_mask"
            )
            self.nan_count = tf.Variable(0, dtype=tf.float32, name="nan_count")
            self.nan_var = np.nan
            # constants to check against
            self.check_const_0 = tf.constant(
                0, dtype=tf.float32, name="const_0"
            )
            self.check_const_1 = tf.constant(
                1, dtype=tf.float32, name="const_1"
            )
            self.check_const_2 = tf.constant(
                2, dtype=tf.float32, name="const_2"
            )
            self.check_const_m1 = tf.constant(
                -1, dtype=tf.float32, name="const_m1"
            )
            # storage variable to evaluate if any skip got triggered. needs
            # to be done without bools, as they are not backward-propagation-
            # compatible.
            # 6 is the number of conditions that are checked
            self.skip_eval = tf.Variable(
                np.array([1 for _ in range(6)]),
                dtype=tf.float32,
                name="skip_eval"
            )

        # init base layer configs
        super().__init__(name=name, **kwargs)

    def call(self, X):
        """heart of the layer: every batch will run through the graph that is
        created here

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        Returns:
            X (tf.Tensor): the batch, manipulated according to the given
                           strategy.
                           shape: [batch_size, seq_len, frame_repr_len]
        """
        if self.strategy == "mp_max_prob":
            """if both hands are labeled as right or left, the one with the
            higher probability will get the position it is labeled with. Else,
            hands will just be sorted according their predicted affiliation
            """
            if self.debug_mode:
                tf.print("IN")
            # call the strategys function for every frame representation. shape
            # at this stage will be [None, sequence_len, csv_coordinate_dims],
            # where None represents the batch size.
            # To recieve the desired operations in a graph compatible way, two
            # tf.map functions are nested. each tf.map function runs over the
            # 0-dim, so the doubled nesting will lead to the transformation of
            # each representation
            X = tf.map_fn(
                fn=(lambda t: tf.map_fn(
                    fn=self._max_prob_decider,
                    elems=t
                )
                    ),
                elems=X
            )
        return (X)

    def _max_prob_decider(self, X):
        """max-probability-strategy function

        Args:
            X (tf.Tensor): part of the batch, of the sequence -> one frame
                           representation

        Returns:
            X (tf.Tensor): the representation with possibly sorted, swapped
                           hands, according to the handedness predictions of
                           mediapipe
        """

        # Condition section.
        # If one of those returns true, X does NOT need to be modified
        # -------------
        def _skip1(hs):
            """check if:
            - no nan's for the hands
            - while h_1 > h_0
            Meaning:
            one left and one right hand is detected, they are already at the
            right places

            Args:
                hs (tf.Tensor): handedness predictions

            Returns:
                (tf.Tensor, dtype=bool): If True, no swap is to be done on this
                                         Tensor
            """
            return (tf.greater(hs[1], hs[0]))

        def _skip2(nan_count, check_const_2):
            """check if:
            - both handedness predicitons contain nan's
            Meaning:
            no hand is detected at all

            Args:
                nan_count (tf.Tensor): counter of nan's at handedness-
                                       predictions (can be 0, 1, 2)
                check_const_2 (tf.Tensor): a constant to check againts (==2)

            Returns:
                (tf.Tensor, dtype=bool): If True, no swap is to be done on this
                                         Tensor
            """
            return (tf.equal(nan_count, check_const_2))

        def _skip3(hs, nan_mask, check_const_1):
            """check if:
            - handedness prediction h_0 == nan
            - handedness prediction h_1 == 1
            Meaning:
            One hand was detected, the coordinates are on hand-position 1
            and the hand detected is detected as right hand

            Args:
                hs (tf.Tensor): handedness predictions
                nan_mask (tf.Tensor): bool mask of nan's at handedness-
                                      predictions
                check_const_1 (tf.Tensor): a constant to check againts (==1)

            Returns:
                (tf.Tensor, dtype=bool): If True, no swap is to be done on this
                                         Tensor
            """
            h_0_is_nan = tf.equal(nan_mask[0], check_const_1)
            h_1_is_1 = tf.equal(hs[1], check_const_1)
            return (tf.math.logical_and(h_0_is_nan, h_1_is_1))

        def _skip4(hs, nan_mask, check_const_0, check_const_1):
            """check if:
            - handedness prediction h_1 == nan
            - handedness prediction h_0 == 0
            Meaning:
            One hand was detected, the coordinates are on hand-position 0
            and the hand detected is detected as left hand

            Args:
                hs (tf.Tensor): handedness predictions
                nan_mask (tf.Tensor): bool mask of nan's at handedness-
                                      predictions
                check_const_0 (tf.Tensor): a constant to check againts (==0)
                check_const_1 (tf.Tensor): a constant to check againts (==1)

            Returns:
                (tf.Tensor, dtype=bool): If True, no swap is to be done on this
                                         Tensor
            """
            h_1_is_nan = tf.equal(nan_mask[1], check_const_1)
            h_0_is_0 = tf.equal(hs[0], check_const_0)
            return (tf.math.logical_and(h_1_is_nan, h_0_is_0))

        def _skip5(hs, hs_prob, nan_count, check_const_0):
            """check if:
            - no nan in hs
            - handedness prediction h_1 == h_0
            - h_n == 0
            - h_0_p > h_1_p
            Meaning:
            Two hands were detected, but both labeled as left hand. As the
            probability of the hand coordinated at position 0 is bigger, this
            hand is interprated as true-left hand and thus, nothing needs to
            be swapped

            Args:
                hs (tf.Tensor): handedness predictions
                hs_prob (tf.Tensor): handedness prediction probabilities
                nan_count (tf.Tensor): counter of nan's at handedness-
                                       predictions (can be 0, 1, 2)
                check_const_0 (tf.Tensor): a constant to check againts (==0)

            Returns:
                (tf.Tensor, dtype=bool): If True, no swap is to be done on this
                                         Tensor
            """
            no_nan = tf.equal(nan_count, check_const_0)
            h_0_equals_h_1 = tf.equal(hs[0], hs[1])
            h_n_equals_0 = tf.equal(hs[0], check_const_0)
            h_0_p_greater = tf.greater(hs_prob[0], hs_prob[1])
            final_eval = tf.math.logical_and(
                tf.math.logical_and(no_nan, h_0_equals_h_1),
                tf.math.logical_and(h_n_equals_0, h_0_p_greater)
            )
            return (final_eval)

        def _skip6(hs, hs_prob, nan_count, check_const_0, check_const_1):
            """check if:
            - no nan in hs
            - handedness prediction h_1 == h_0
            - h_n == 1
            - h_0_p < h_1_p
            Meaning:
            Two hands were detected, but both labeled as right hand. As the
            probability of the hand coordinates at position 1 is bigger, this
            hand is interprated as true-right hand and thus, nothing needs to
            be swapped

            Args:
                hs (tf.Tensor): handedness predictions
                hs_prob (tf.Tensor): handedness prediction probabilities
                nan_count (tf.Tensor): counter of nan's at handedness-
                                       predictions (can be 0, 1, 2)
                check_const_0 (tf.Tensor): a constant to check againts (==0)
                check_const_1 (tf.Tensor): a constant to check againts (==1)

            Returns:
                (tf.Tensor, dtype=bool): If True, no swap is to be done on this
                                         Tensor
            """
            no_nan = tf.equal(nan_count, check_const_0)
            h_0_equals_h_1 = tf.equal(hs[0], hs[1])
            h_n_equals_1 = tf.equal(hs[0], check_const_1)
            h_1_p_greater = tf.greater(hs_prob[1], hs_prob[0])
            final_eval = tf.math.logical_and(
                tf.math.logical_and(no_nan, h_0_equals_h_1),
                tf.math.logical_and(h_n_equals_1, h_1_p_greater)
            )
            return (final_eval)

        # initialise variables on the fly acording to the current instance
        # get handedness results and probabilities
        self.hs.assign([X[self.pos_0_idx], X[self.pos_1_idx]])
        self.hs_prob.assign([X[self.prob_0_idx], X[self.prob_1_idx]])

        if self.debug_mode:
            tf.print("tensor_inp: ", self.hs, self.hs_prob)

        # mask -1's with nan's. This is due to a later change in how not
        # detected hands are marked in the data input, which was originally done
        # with nan's, but later changed to -1. This workaround easily ensures
        # this layer continues to work properly, without needing to change
        # almost every function in here. But of course, at some point this would
        # be the clearner and faster way to do it. Even though the speed
        # difference in graph-mode should be marginal
        self.hs.assign(
            tf.where(self.hs < 0, self.nan_var, self.hs)
        )
        self.hs_prob.assign(
            tf.where(self.hs_prob < 0, self.nan_var, self.hs_prob)
        )

        # print handedness results for every frame representation
        if self.debug_mode:
            tf.print("transf_inp: ", self.hs, self.hs_prob)

        # create nan mask for handedness results
        self.nan_mask.assign(
            tf.dtypes.cast(
                tf.math.is_nan(self.hs),
                dtype=tf.float32)
        )
        # accumulate nan mask
        self.nan_count.assign(
            tf.math.reduce_sum(self.nan_mask)
        )

        # perform tests against the skip conditions
        # ------------------
        # SKIP 1 - description at the function
        self.skip_eval[0].assign(tf.cast(
            _skip1(
                self.hs
            ),
            tf.float32
        )
        )
        # SKIP 2 - description at the function
        self.skip_eval[1].assign(tf.cast(
            _skip2(
                self.nan_count,
                self.check_const_2
            ),
            tf.float32
        )
        )
        # SKIP 3 - description at the function
        self.skip_eval[2].assign(tf.cast(
            _skip3(
                self.hs,
                self.nan_mask,
                self.check_const_1
            ),
            tf.float32
        )
        )
        # SKIP 4 - description at the function
        self.skip_eval[3].assign(tf.cast(
            _skip4(
                self.hs,
                self.nan_mask,
                self.check_const_0,
                self.check_const_1
            ),
            tf.float32
        )
        )
        # SKIP 5 - description at the function
        self.skip_eval[4].assign(tf.cast(
            _skip5(
                self.hs,
                self.hs_prob,
                self.nan_count,
                self.check_const_0
            ),
            tf.float32
        )
        )
        # SKIP 6 - description at the function
        self.skip_eval[5].assign(tf.cast(
            _skip6(
                self.hs,
                self.hs_prob,
                self.nan_count,
                self.check_const_0,
                self.check_const_1
            ),
            tf.float32
        )
        )

        # functions to process the result from skip_eval. use local vars inside
        # the functions, as they can only be called without arguments
        # pass func
        def _pass_func():
            """helper function for the conditional swapper: return the original
            X, as this function is triggered if nothing needs to be swapped

            Returns:
                X (tf.Tensor): locally available, swapped X, which is a singel
                               frame representation)
            """
            if self.debug_mode:
                tf.print("PASS")
            return (X)

        # swap function
        def _swap():
            """helper function for the conditional swapper: sort/swap X and
            return the sorted/swapped X

            Returns:
                X (tf.Tensor): locally available, swapped X, which is a singel
                               frame representation)
            """
            # signal that a swap was performed. can be correlated to the
            # handedness results that are also printed in debug mode
            if self.debug_mode:
                tf.print("SWAP")
            # CHECK: does the shape-tensor create a new graph every time??
            # get the shape, swap the tensor
            shape = tf.shape(X, out_type=tf.int64)
            scattered_tensor = tf.scatter_nd(self.swap_pattern, X, shape)
            return (scattered_tensor)

        # finally evaluate the skip checks, return original X if any (or more)
        # of the skip test retuned True. If no skip returned True, swap the
        # hands and return the swapped X
        X = tf.cond(
            tf.math.reduce_sum(self.skip_eval) > self.check_const_0,
            _pass_func,
            _swap
        )
        return (X)

    def compute_output_shape(self, input_shape):
        """this function is needed to seemingly use the custom layer

        Args:
            input_shape
        Returns:
            input_shape
        """
        return (input_shape)

    def get_config(self):
        """this function is needed to seemingly use the custom layer

        Returns:
            _configs (dict): an assignment of all variables that were used to
                             initialise the layer
        """
        # get and safe params used
        config = {
            "strategy": self.strategy,
            "csv_struct": self.csv_struct,
            "debug_mode": self.debug_mode
        }
        base_config = super().get_config()
        return (dict(list(base_config.items()) + list(config.items())))


# decorator ensures easy loading of the custom layers
@keras.utils.register_keras_serializable(package="Custom", name="Hand_Imputer")
class Hand_Imputer(keras.layers.Layer):
    """imputation layer for missing hand coordinates
    """

    def __init__(self, strategy="dummy_vals", debug_mode=False,
                 name="Hand_Imputer", **kwargs):
        """initialise everything needed for this layer

        Args:
            strategy (str): in ["dummy_vals"]
            debug_mode (bool): activates some prints if enabled
            name (str): name of the layer. defaults to tha class name
        """
        # save strategy and debug-state
        self.strategy = strategy
        self.debug_mode = debug_mode
        # initialise the dummy value that is used to fill nan's if the dummy_val
        # strategy is used
        self.dummy_val = tf.constant(2.)

        # init base layer configs
        super().__init__(name=name, **kwargs)

    def call(self, X):
        """heart of the layer: every batch will run through the graph that is
        created here

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        Returns:
            X (tf.Tensor): the batch, manipulated according to the given
                           strategy.
                           shape: [batch_size, seq_len, frame_repr_len]
        """
        if self.debug_mode:
            tf.print("\n\n\nENTERING imputer")

        # NOTE: this strategy assumes only NaN's from missing hand data are left
        #		as simply all NaN's are imputed
        if self.strategy == "dummy_vals":
            if self.debug_mode:
                tf.print("imputing strategy: DUMMY_VALS")
                tf.print("imputing nan's with: ", self.dummy_val)

            # mask nan's with bool's
            nan_mask_bool = tf.math.is_nan(X)
            # create tensor masks that are usable for matrix multiplications
            mask = tf.dtypes.cast(
                nan_mask_bool,
                dtype=tf.float32
            )
            inverse_mask = tf.dtypes.cast(
                tf.math.logical_not(nan_mask_bool),
                dtype=tf.float32
            )
            # convert all nan's to zeros, leave the rest untouched
            zeroed = tf.math.multiply_no_nan(X, inverse_mask)
            # create an adder-tensor
            add_mask = tf.math.scalar_mul(self.dummy_val, mask)

            # give a little insight BEFORE imputation
            if self.debug_mode:
                tf.print("\nbefore imputation (showing a slice)")
                tf.print(X[:, :, 3:10])

            # finaly combine everything, sothat nan's become the dummy value
            X = tf.add(zeroed, add_mask)

            # give a little insight AFTER imputation
            if self.debug_mode:
                tf.print("\nafter imputation (showing a slice)")
                tf.print(X[:, :, 3:10])
        else:
            # signal if no strategy got triggered, possibly due to a wrongly
            # named strategy
            tf.print("\nWARNING! (Hand_Imputer)\nNo strategy got triggered by" \
                     " given keyword.\nBatch is passed without imputing-" \
                     "transformation")
        return (X)

    def compute_output_shape(self, input_shape):
        """this function is needed to seemingly use the custom layer

        Args:
            input_shape
        Returns:
            input_shape
        """
        return (input_shape)

    def get_config(self):
        """this function is needed to seemingly use the custom layer

        Returns:
            _configs (dict): an assignment of all variables that were used to
                             initialise the layer
        """
        config = {
            "strategy": self.strategy,
        }
        base_config = super().get_config()
        return (dict(list(base_config.items()) + list(config.items())))


# decorator ensures easy loading of the custom layers
@keras.utils.register_keras_serializable(package="Custom", name="NaNs_to_zero")
class NaN_to_zero(keras.layers.Layer):
    """layer will convert all NaN's in a given instance/batch to 0's
    """

    def __init__(self, name="NaNs_to_zero", **kwargs):
        """initialise everything needed for this layer, which is basicly nothing
        """
        super().__init__(name=name, **kwargs)

    def call(self, X):
        """heart of the layer: every batch will run through the graph that is
        created here

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        Returns:
            X (tf.Tensor): the batch with nan's converted to 0
                           shape: [batch_size, seq_len, frame_repr_len]
        """
        # mask nan values
        value_not_nan = tf.dtypes.cast(
            tf.math.logical_not(
                tf.math.is_nan(X)
            ),
            dtype=tf.float32
        )
        # this tf method return 0 for X if value_not_nan is 0, no matter what
        # X is originally. As this also works if X is nan, this method works
        X = tf.math.multiply_no_nan(X, value_not_nan)
        return (X)

    def compute_output_shape(self, input_shape):
        """this function is needed to seemingly use the custom layer

        Args:
            input_shape
        Returns:
            input_shape
        """
        return (input_shape)

    def get_config(self):
        """this function is needed to seemingly use the custom layer

        Returns:
            _configs (dict): an assignment of all variables that were used to
                             initialise the layer
        """
        config = {
            # nothing (yet)
        }
        base_config = super().get_config()
        return (dict(list(base_config.items()) + list(config.items())))


# decorator ensures easy loading of the custom layers
@keras.utils.register_keras_serializable(package="Custom", name="Normalizer")
class Normalizer(keras.layers.Layer):
    """provide different normalization strategies, which will not only ensure
    normalized data, but also how general a net, trained with a choosen
    strategy, can opereate.
    The strategies provided are listed from very setup&case specific to most
    general. Note that for a general approach, the dataset used for training
    should use a bunch of randomized data augmentation technics.

    Strategy 1 (pic_bound):
        Normalize correlated to the x-y space of the given of the picture.

    Strategy 2 (seq_on_last):
        Normalize coordinates (seperated per hand) to the wrist points of the
        most current point which is available for this hand

    Strategy 3 (inst_on_self):
        Normalize all coordinates related to themselves, sothat they appear
        stational.
    """

    def __init__(self, strategy="pic_bound", csv_struct="v2", name="Normalizer",
                 debug_mode=False, **kwargs):
        """initialise everything needed for this layer

        Args:
            strategy (str): in ["pic_bound", "seq_on_last", "inst_on_self"]
            csv_struct (str): the structure of the used data. in ["v2"]
            name (str): name of the layer. defaults to tha class name
            debug_mode (bool): activates some prints if enabled
        """
        # save strategy, structure and debug-state
        self.strategy = strategy
        self.debug_mode = debug_mode
        self.csv_struct = csv_struct

        # to properly work with numerical zeros
        self.eps = 1e-8

        if self.csv_struct == "v2":
            # structure object for csv-file-based-structure version 2
            self.arch = cpa.CSV_Architecture_v2()

        # base vector that will be used to genereate the vector per instance
        # which will be subtracted to normalize the hands
        self.substract_vec_base = tf.constant(
            np.array([0 for _ in range(len(self.arch.assignment))]),
            dtype=tf.float32
        )

        # initialise the swap pattern which is used by the normalize-on-self
        # approach to rearange the instance-tensors into theire original
        # order
        swap_pattern_np = np.concatenate(
            [
                self.arch.vector_intro_idxs,
                [self.arch.handedness_0_idx, self.arch.handedness_0_prob_idx],
                self.arch.hand_0_x,
                self.arch.hand_0_y,
                self.arch.hand_0_z,
                [self.arch.handedness_1_idx, self.arch.handedness_1_prob_idx],
                self.arch.hand_1_x,
                self.arch.hand_1_y,
                self.arch.hand_1_z,
            ]
        )
        self.swap_pattern = tf.constant(np.expand_dims(swap_pattern_np, -1))

        # init base layer configs
        super().__init__(name=name, **kwargs)

    def call(self, X):
        """heart of the layer: every batch will run through the graph that is
        created here

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        Returns:
            X (tf.Tensor): the batch normalized according to the given strategy
                           shape: [batch_size, seq_len, frame_repr_len]
        """
        # at this stage will be [None, sequence_len, csv_coordinate_dims],
        # where None represents the batch size.
        if self.debug_mode:
            tf.print("\nin normalizer")
            print("also eager in normalizer")

        # build the graphs, according to the given strategy
        if self.strategy == "pic_bound":
            X = self._pic_bound_transformer(X)
        elif self.strategy == "seq_on_last":
            X = tf.map_fn(
                fn=self._seq_on_last_transformer,
                elems=X
            )
        elif self.strategy == "inst_on_self":
            X = tf.map_fn(
                fn=(lambda t: tf.map_fn(
                    fn=self._inst_on_self_transformer,
                    elems=t
                )
                    ),
                elems=X
            )
        else:
            # signal if no strategy got triggered, possibly due to a wrongly
            # named strategy
            tf.print("\nWARNING! (Normalizer)\nNo strategy got triggered by" \
                     " given keyword.\nBatch is passed without normalization-" \
                     "transformation")
        return (X)

    def _pic_bound_transformer(self, X):
        """Strategy 1 (pic_bound):
        Normalize correlated to the x-y space of the given of the picture.
        -> This is just the output from mediapipe: from 0 to 1 for each
           dimension, related to the frame. That also means, for this strategy,
           nothing more needs to be done

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]

        Returns:
            X (tf.Tensor): just the same as the given X, nothing to do here
        """
        if self.debug_mode:
            tf.print(
                "nothing to do in mapped method '_pic_bound_transformer()'"
            )
        return (X)

    @tf.function
    def _seq_on_last_transformer(self, X):
        """Strategy 2 (seq_on_last):
        Normalize coordinates of the complete sequence (seperated per hand) to
        the wrist points of the most current point which is available for this
        hand

        Args:
            X (tf.Tensor): a sequence of the batch
                           shape: [seq_len, frame_repr_len]

        Returns:
            X (tf.Tensor): the sequence, normalized per hand, to the most
                           current, available wrist point
        """
        # NOTE: this approach is tightly bound to csv-structure 2. therefore
        # 		provide a print if anyone in the future will use this with
        # 		another csv-architcture
        if self.csv_struct != "v2":
            tf.print("\n\nWarning! Using Normalize-layer with\n" \
                     "_seq_on_last_transformer and another csv-structure\n" \
                     "than v2. This might lead to unexpacted behaviour when\n" \
                     "the wrist points are gathered! It is recommended to\n" \
                     "validate the layers behaviour and to look up the\n" \
                     "source code.")
        if self.debug_mode:
            tf.print("\n\n\n      in transformer: SEQ_ON_LAST      ")
            tf.print("-- FOR MAXIMUM INFORMATION, DISSABLE --\n" \
                     "-- @tf.function DECORATOR AND ENABLE --\n" \
                     "--          EAGER EXECUTION          --")

        # this reference is only used if a debug print is requested. assign
        # it here anyway as a conditional assignment only in debug mode violates
        # graph rules
        X_init = X

        # get the wrist coordiantes across all frames of the instance
        wrists_0 = tf.gather(
            X,
            [
                self.arch.hand_0_x[0],
                self.arch.hand_0_y[0],
                self.arch.hand_0_z[0]
            ],
            axis=-1
        )
        wrists_1 = tf.gather(
            X,
            [
                self.arch.hand_1_x[0],
                self.arch.hand_1_y[0],
                self.arch.hand_1_z[0]
            ],
            axis=-1
        )

        # DEPCRACATED approach
        # mask where hands were detected:
        # based on the std along the frame-wirsts-axis
        if False:
            wrists_0_stds = tf.math.reduce_std(wrists_0, axis=-1)
            wrists_1_stds = tf.math.reduce_std(wrists_1, axis=-1)
            wrists_0_mask = tf.greater(wrists_0_stds, self.eps)
            wrists_1_mask = tf.greater(wrists_1_stds, self.eps)

        # NEW approach
        # mask where hands were detected:
        # based on the [...]_pred-mark in the coordinate representation
        hand_0_pred = X[:, self.arch.handedness_0_idx]
        hand_1_pred = X[:, self.arch.handedness_1_idx]
        wrists_0_mask = tf.not_equal(
            hand_0_pred,
            self.arch.no_hand_detected_val
        )
        wrists_1_mask = tf.not_equal(
            hand_1_pred,
            self.arch.no_hand_detected_val
        )

        # get indexes of detected hands
        wrists_0_det = tf.where(wrists_0_mask)[:, 0]
        wrists_1_det = tf.where(wrists_1_mask)[:, 0]

        # get latest detected hand, predifine to enable seamless AutoGraph use
        # substract it's wrist coordinate values
        # go seperated path for both hands and concatenate later, as the
        # detection of one hand has to do nothing with the detection of the
        # other one
        first_valid_wrist_idx_0 = tf.constant(-1, dtype=tf.int64)
        substract_vec_0 = self.substract_vec_base
        # pre initialise for autograph compatiblity, even as the matrix later
        # gets created by scattering the subtaction vector
        # repeat also needs to be done here, as the seq-len is unknown during
        # graph creation
        subtraction_matrix_0 = tf.repeat(
            tf.expand_dims(substract_vec_0, axis=0),
            repeats=tf.shape(X)[0],
            axis=0
        )
        # repeat the steps above for hand 1
        first_valid_wrist_idx_1 = tf.constant(-1, dtype=tf.int64)
        substract_vec_1 = self.substract_vec_base
        subtraction_matrix_1 = tf.repeat(
            tf.expand_dims(substract_vec_1, axis=0),
            repeats=tf.shape(X)[0],
            axis=0
        )

        # print first useful information in debug mode
        if self.debug_mode:
            tf.print("\nprocessed sequence (squeezed display):\n", X[:, 3:])
            tf.print("shape: ", X.shape)
            tf.print("positions_valid_hands hand_0: ", wrists_0_det)
            tf.print("positions_valid_hands hand_1: ", wrists_1_det)

        # only enter if there is at least one valid hand. build the subtraction
        # matrix and subtract it from the instance
        # NOTE: build the two matrices - one for each hand - based on a
        #		0-initialised matrix and add them together to the final
        #		subtraction matrix
        # hand_0
        if tf.not_equal(tf.shape(wrists_0_det)[0], 0):
            # reinitialise with same datatype as at local definition
            first_valid_wrist_idx_0 = wrists_0_det[-1]
            ref_wrist = tf.expand_dims(
                wrists_0[first_valid_wrist_idx_0],
                axis=1
            )
            # combine the indexes according to the dimension
            idx_s_comb = ([
                self.arch.hand_0_x,
                self.arch.hand_0_y,
                self.arch.hand_0_z
            ])
            # build the vector that shall be substracted from every valid hand
            subtraction_matrix_0 = self._get_sub_matrix_seq_on_last(
                ref_wrist,
                idx_s_comb,
                substract_vec_0,
                wrists_0_det,
                tf.shape(X)
            )
            if self.debug_mode:
                tf.print("CREATED sub_matr for hand_0")
        else:
            if self.debug_mode:
                tf.print("NO sub_matr created for hand_0, due to no valid " \
                         "hand in seq")
        # hands_1
        if tf.not_equal(tf.shape(wrists_1_det)[0], 0):
            # reinitialise with same datatype as at local definition
            first_valid_wrist_idx_1 = wrists_1_det[-1]
            ref_wrist = tf.expand_dims(
                wrists_1[first_valid_wrist_idx_1],
                axis=1
            )
            # combine the indexes according to the dimension
            idx_s_comb = ([
                self.arch.hand_1_x,
                self.arch.hand_1_y,
                self.arch.hand_1_z
            ])
            # build the vector that shall be substracted from every valid hand
            subtraction_matrix_1 = self._get_sub_matrix_seq_on_last(
                ref_wrist,
                idx_s_comb,
                substract_vec_1,
                wrists_1_det,
                tf.shape(X)
            )
            if self.debug_mode:
                tf.print("CREATED sub_matr for hand_1")
        else:
            if self.debug_mode:
                tf.print("NO sub_matr created for hand_1, due to no valid " \
                         "hand in seq")

        # combine the subtraction matrices that were created seperatly per hand,
        # but across all frames representated by the instance
        subtraction_matrix = tf.math.add(
            subtraction_matrix_0,
            subtraction_matrix_1
        )

        if self.debug_mode:
            tf.print("\nhand_0 snip BEFORE trans")
            tf.print(X[:, 4:9])

        # finally process the instance
        X = tf.math.subtract(X, subtraction_matrix)

        if self.debug_mode:
            tf.print("\nhand_0 snip AFTER trans")
            tf.print(X[:, 4:9])

        # this print will show a hugh output, but it will only print if the
        # @tf.function-decorator is uncommented and the model runs eagerly
        if self.debug_mode:
            print("\ninput sequence: ")
            print(X_init)
            print("\ntransformed sequence: ")
            print(X)
            print("\nused subtraction matrix: ")
            print(subtraction_matrix)

        return (X)

    @tf.function
    def _inst_on_self_transformer(self, X_t):
        """Strategy 3 (inst_on_self):
        Normalize all coordinates related to themselves, sothat they appear
        stational. Therefore, each framerepresentation needs to be normalized
        in relation to itself, splitted by hands and dimensions

        FLOW:
        - extract both hand's coordinate's seperate
        - check if a valid hand is there
        - normalize on wrist per hand
        - restack the normalized hands with the other parameters to the return
          value

        Args:
            X (tf.Tensor): a fram representation of a sequence of the batch
                           shape: [frame_repr_len]

        Returns:
            X (tf.Tensor): the representation, normalized per hand and
                           dimension, but only for valid hands (do nothing for
                           not detected hands)
        """
        # NOTE: the input is every frame representation itself and not the
        # 		whole instance (which would mean be sequence of representations)
        # NOTE: this approach is tightly bound to csv-structure 2. therefore
        # 		provide a print if anyone in the future will use this with
        # 		another csv-architcture
        if self.csv_struct != "v2":
            tf.print("\n\nWarning! Using Normalize-layer with\n" \
                     "_inst_on_self_transformer and another csv-structure\n" \
                     "than v2. This might lead to unexpacted behaviour when\n" \
                     "the values are restacked! It is recommended to\n" \
                     "validate the layers behaviour and to look up the\n" \
                     "source code.")
        if self.debug_mode:
            tf.print("\n\n\n      in transformer: INST_ON_SELF      ")
            tf.print("-- FOR MAXIMUM INFORMATION, DISSABLE --\n" \
                     "-- @tf.function DECORATOR AND ENABLE --\n" \
                     "--          EAGER EXECUTION          --")

        # this reference is only used if a debug print is requested. assign
        # it here anyway as a conditional assignment only in debug mode violates
        # graph rules
        X_init = X_t

        # get the handedness prediction per hand
        X_t_hand_0_pred = X_t[self.arch.handedness_0_idx]
        X_t_hand_1_pred = X_t[self.arch.handedness_1_idx]

        # assign some stuff here, to not do it within the conditions, to easaly
        # and smoothly enable autograph
        # collect the coordinates per dimension per hand
        # init values
        X_t_init_vals = tf.gather(
            X_t,
            self.arch.vector_intro_idxs.tolist(),
            axis=-1
        )
        # hand_0
        X_t_hand_0_x = tf.gather(
            X_t,
            self.arch.hand_0_x.tolist(),
            axis=-1
        )
        X_t_hand_0_y = tf.gather(
            X_t,
            self.arch.hand_0_y.tolist(),
            axis=-1
        )
        X_t_hand_0_z = tf.gather(
            X_t,
            self.arch.hand_0_z.tolist(),
            axis=-1
        )
        # hand_1
        X_t_hand_1_x = tf.gather(
            X_t,
            self.arch.hand_1_x.tolist(),
            axis=-1
        )
        X_t_hand_1_y = tf.gather(
            X_t,
            self.arch.hand_1_y.tolist(),
            axis=-1
        )
        X_t_hand_1_z = tf.gather(
            X_t,
            self.arch.hand_1_z.tolist(),
            axis=-1
        )

        # print first useful information in debug mode
        if self.debug_mode:
            tf.print(
                "\nprocessed frame representation (squeezed display):\n",
                X_t
            )
            tf.print("shape: ", X_t.shape)
            tf.print("handedness_pred hand_0: ", X_t_hand_0_pred)
            tf.print("handedness_pred hand_1: ", X_t_hand_1_pred)

        # hand_0: modifiy if there is a valid hand
        if tf.not_equal(X_t_hand_0_pred, self.arch.no_hand_detected_val):
            if self.debug_mode:
                tf.print("IN for hand_0: normalize coordinates per dimension")
            # normalize all dimensions by themselves. use the preinitialised
            # tensors
            X_t_hand_0_x = self._min_max_0_1(X_t_hand_0_x)
            X_t_hand_0_y = self._min_max_0_1(X_t_hand_0_y)
            X_t_hand_0_z = self._min_max_0_1(X_t_hand_0_z)
        else:
            if self.debug_mode:
                tf.print("NOT_in for hand_0: no valid hand detected, pass hand")
        # hand_1: modifiy if there is a valid hand
        if tf.not_equal(X_t_hand_1_pred, self.arch.no_hand_detected_val):
            if self.debug_mode:
                tf.print("IN for hand_1: normalize coordinates per dimension")
            # normalize all dimensions by themselves. use the preinitialised
            # tensors
            X_t_hand_1_x = self._min_max_0_1(X_t_hand_1_x)
            X_t_hand_1_y = self._min_max_0_1(X_t_hand_1_y)
            X_t_hand_1_z = self._min_max_0_1(X_t_hand_1_z)
        else:
            if self.debug_mode:
                tf.print("NOT_in for hand_1: no valid hand detected, pass hand")

        # this print will show a hugh output, but it will only print if the
        # @tf.function-decorator is uncommented and the model runs eagerly
        if self.debug_mode:
            print("\ninitial look of the frame representation:\n")
            print(X_t)

        # restack the instance tensor with the possibly modified sequences
        X_t = tf.concat(
            [
                X_t_init_vals,
                tf.expand_dims(X_t[self.arch.handedness_0_idx], 0),
                tf.expand_dims(X_t[self.arch.handedness_0_prob_idx], 0),
                X_t_hand_0_x,
                X_t_hand_0_y,
                X_t_hand_0_z,
                tf.expand_dims(X_t[self.arch.handedness_1_idx], 0),
                tf.expand_dims(X_t[self.arch.handedness_1_prob_idx], 0),
                X_t_hand_1_x,
                X_t_hand_1_y,
                X_t_hand_1_z
            ],
            0
        )

        # this print will show a hugh output, but it will only print if the
        # @tf.function-decorator is uncommented and the model runs eagerly
        if self.debug_mode:
            print("\nrestacked instance with possibly normalized values:\n")
            print(X_t)
            print("\ncurr DIFF to init-inst (for val):")
            print(" NOTE: as the values got normalized, the diff tensor is" \
                  "\n not equal to zero, despite that values are at the right" \
                  "\n index. Deactivate the normalize-function calls within" \
                  "\n this layers function, to validate the correct swapping")
            print(tf.subtract(X_init, X_t))

        # swap the values back into original order
        shape = tf.shape(X_t, out_type=tf.int64)
        X_t = tf.scatter_nd(self.swap_pattern, X_t, shape)

        # this print will show a hugh output, but it will only print if the
        # @tf.function-decorator is uncommented and the model runs eagerly
        if self.debug_mode:
            print("\ntensor after scattering with swap pattern (this one\n" \
                  "is finally returned")
            print(X_t)
            print("\ncurr diff to init inst (for val):")
            print(" NOTE: as the values got normalized, the diff tensor is" \
                  "\n not equal to zero, despite that values are at the right" \
                  "\n index. Deactivate the normalize-function calls within" \
                  "\n this layers function, to validate the correct swapping")
            print(tf.subtract(X_init, X_t))

        return (X_t)

    def _min_max_0_1(self, X_seq):
        """normalize a given sequence from X on itself (min/max to 0/1)

        Args:
            X_seq (tf.Tensor): a sequence to normalize onto 0-1. While writing,
                               the input is always a slice from one hand and
                               one dimension, e.g. all x-coordinates from hand_0
                               shape: [seq_len]

        Returns:
            X_seq (tf.Tensor): the normalized input sequence
        """
        X_seq = tf.math.divide(
            tf.math.subtract(
                X_seq,
                tf.math.reduce_min(X_seq)
            ),
            tf.add(
                tf.math.subtract(
                    tf.math.reduce_max(X_seq),
                    tf.math.reduce_min(X_seq)
                ),
                self.eps
            )
        )
        return (X_seq)

    def _get_sub_matrix_seq_on_last(self, ref_wrist, idx_s_comb,
                                    substract_vec, wrists_det, X_shape):
        """provide the matrix which is used by the _seq_on_last_transformer-
        function to normalize a sequence of frame representations. the function
        builds a matrix, which's vectors repeat the x,y&z-coordinate of the
        wrist point from the most current hand. The coordinates get placed at
        the right indexes for dimension&hand; unvalid hands are also already
        concidered and simply reciev zeros.
        The function is called one per hand and the returned matrix will contain
        zeros for the other hand.
        The matrices genereated that way, will than be added, and the resulting
        matrix can simply get subracted from the instance (which is done in
        the _seq_on_last_transformer itself)

        Args:
            ref_wrist (tf.Tensor): the wirst coordinates (x,y,z) of the wrist
                                   point on which the whole sequence shall be
                                   normalized
            idx_s_comb (list(np.array)): contains three numpy arrays, which
                                         contain all indexes for one dimension
                                         in the current frame representation
            substract_vec (tf.Tensor): this vector will contain the x,y and z
                                       values at the right indexes, for one
                                       hand. This vector is to be subtracted
                                       from valid hands
            wrists_det (tf.Tensor): contains the indexes of the validly detected
                                    hands. It is needed to know how often and
                                    where the substract_vec needs to be inserted
            X_shape (tf.): shape of an instance, which is
                           [seq_len, frame_repr_len]. The shape is needed to
                           create the subtraction_matrix, as the tf.scatter_nd-
                           function wants an explicit shape

        Returns:
            subtraction_matrix (tf.Tensor): the matrix which contains the right
                                            values at the right position to
                                            normalize one hand of an instance.
                                            To normalize the whole instance,
                                            this function needs to be called per
                                            hand, the recieved matrices combined
                                            and this final matrix can then be
                                            subtracted from the whole instance
        """
        # loop through the coordinate points. work with return values according
        # to the autograph guidlines
        subtract_vec = substract_vec
        # itterate over x, y and z dimension. NOTE: cannot use zip function
        # with autograph, therefore use index assignment
        for i in range(len(idx_s_comb)):
            # assign the x/y/z-value which is used for normalization across all
            # frame representations, as well as the indexes for each frame that
            # shall be affected by it
            ref_val, idx_s = ref_wrist[i], idx_s_comb[i]

            # expand dims to properly use the array with tf's scatter_nd
            # function
            val_idx = tf.convert_to_tensor(
                np.expand_dims(idx_s, 1),
                dtype=tf.int32
            )

            # create repeated vec that will be the coordinat of the current
            # wrist point) whichs contains the coordinate the number of times
            # it will be imputed into the substraction vector
            rep_val = tf.repeat(ref_val, repeats=tf.shape(idx_s)[0])

            # scattered inputs
            subtract_vec = tf.tensor_scatter_nd_update(
                subtract_vec,
                val_idx,
                rep_val
            )

        # scatter the vector according to the instance size and according to
        # the properly detected hands
        rep_sub_vec = tf.repeat(
            tf.expand_dims(subtract_vec, axis=0),
            repeats=tf.shape(wrists_det)[0],
            axis=0
        )

        # create the subtraction matrix for the given hand. NOTE: this matrix
        # needs to be combined with the one for the other hand, before it is
        # finally applied to the instance
        subtraction_matrix = tf.scatter_nd(
            tf.expand_dims(wrists_det, axis=1),
            rep_sub_vec,
            shape=tf.cast(X_shape, tf.int64)
        )
        return (subtraction_matrix)

    def compute_output_shape(self, input_shape):
        """this function is needed to seemingly use the custom layer

        Args:
            input_shape
        Returns:
            input_shape
        """
        return (input_shape)

    def get_config(self):
        """this function is needed to seemingly use the custom layer

        Returns:
            _configs (dict): an assignment of all variables that were used to
                             initialise the layer
        """
        config = {
            "strategy": self.strategy,
            "csv_struct": self.csv_struct,
            "debug_mode": self.debug_mode
        }
        base_config = super().get_config()
        return (dict(list(base_config.items()) + list(config.items())))


# decorator ensures easy loading of the custom layers
@keras.utils.register_keras_serializable(package="Custom", name="Man_Dim_Red")
class Manual_Dim_Reducer(keras.layers.Layer):
    """manage the inputs, derived from inf1. Key functionalitys are the handling
    of the probability's given for the hands, handling NaNs that are NOT from
    coordinate columns and perform simple dimensionality reduction

    NOTE: no matter which strategy is choosen, timestamp are always removed as
          they are complety unrelated to the classification
    NOTE: "[...]_full" always means, additional informations like handedness
          results are kept, "[...]_red means" the tensor gets reduced to not
          contain them
    NOTE: the handedness prediction contains two values per hand: which hand
          it is, and how certain mediapipe is about the hand prediction. The
          indicating value which hand it is could be dropped, as dependend on
          this value, the handcoordinates are swapped. Keep it in anyway, as
          for different imputation strategies the value might indicate different
          things, as well as the signal that no hand was detected

    strategy in: [
        "_full"				: do nothing special
        "_red"				: drop additional informations etc.
        "del_z_full"		: delete z-dimensionl, keep additioanl info
        "del_z_red"			: delete z-dimension and additional infos
        "avg_of_dims_full"	: average over all points per dimension/hand
        "avg_of_dims_red"	: average over all points per dimension/hand
        "sig_5_points_full"	: keep only 5 significant points per hand
        "sig_5_points_red"	: keep only 5 significant points per hand
        ""
    ]
    """

    def __init__(self, strategy="del_z_red", csv_struct="v2", debug_mode=False,
                 name="Man_Dim_Red", **kwargs):
        """initialise everything needed for this layer

        Args:
            strategy (str): in ["dummy_vals"]
            csv_struct (str): the structure of the used data. in ["v2"]
            debug_mode (bool): activates some prints if enabled
            name (str): name of the layer. defaults to tha class name
        """
        # save strategy, structure and debug-state
        self.strategy = strategy
        self.debug_mode = debug_mode
        self.csv_struct = csv_struct

        # initialise the data-structure object
        if self.csv_struct == "v2":
            # structure object for csv-file-based-structure version 2
            self.arch = cpa.CSV_Architecture_v2()

        # will be set to false if the strategy doesnt trigger one of the
        # following conditions, which means, no slicing strategy will take
        # place, which will be told to the user if the debug mode is activated
        self.slicing_strategy_initialised = True

        # initialise tensors with the indexes to slice out from the incoming
        # instances. This is of course strategy dependant and also not done
        # for every strategy, as some also rely on calculation and not only
        # slicing
        if self.strategy == "_full":
            # get all indexes according to the csv structure
            len_arch = len(self.arch.assignment)
            all_idxs = np.array([i for i in range(len_arch)])
            # drop the aditional information indexes (like timestamps etc.)
            rel_idxs_np = np.delete(all_idxs, self.arch.vector_intro_idxs)
            # convert them to a tensor for easy use with tf.gather later
            self.rel_idxs = tf.constant(rel_idxs_np, dtype=tf.int32)
        elif self.strategy == "_red":
            # combine the hand related indexes, without handedness information
            # etc.
            rel_idxs_np = np.concatenate(
                (self.arch.hand_0_idxs_coo_only,
                 self.arch.hand_1_idxs_coo_only)
            )
            # convert them to a tensor for easy use with tf.gather later
            self.rel_idxs = tf.constant(rel_idxs_np, dtype=tf.int32)
        elif self.strategy == "del_z_full":
            # get all indexes according to the csv structure
            len_arch = len(self.arch.assignment)
            all_idxs = np.array([i for i in range(len_arch)])
            # drop the aditional information indexes (like timestamps etc.)
            all_idxs_clean = np.delete(all_idxs, self.arch.vector_intro_idxs)
            # mask all z-dim-related indexes as True
            all_hand_idxs_w_masked_z = np.in1d(
                all_idxs_clean,
                np.concatenate((
                    self.arch.hand_0_z,
                    self.arch.hand_1_z)
                )
            ).reshape(all_idxs_clean.shape)
            # delete values at the indexes where the mask is True -> all z-
            # related-coordinates
            rel_idxs_np = np.delete(
                all_idxs_clean,
                np.where(all_hand_idxs_w_masked_z)
            )
            # convert them to a tensor for easy use with tf.gather later
            self.rel_idxs = tf.constant(rel_idxs_np, dtype=tf.int32)
        elif self.strategy == "del_z_red":
            # collect all coordinate indexes, without handedness information
            # etc.
            all_hand_idxs = np.concatenate(
                (self.arch.hand_0_idxs_coo_only,
                 self.arch.hand_1_idxs_coo_only)
            )
            # mask all z-dim-related indexes as True
            all_hand_idxs_w_masked_z = np.in1d(
                all_hand_idxs,
                np.concatenate((
                    self.arch.hand_0_z,
                    self.arch.hand_1_z)
                )
            ).reshape(all_hand_idxs.shape)
            # delete values at the indexes where the mask is True -> all z-
            # related-coordinates
            rel_idxs_np = np.delete(
                all_hand_idxs,
                np.where(all_hand_idxs_w_masked_z)
            )
            # convert them to a tensor for easy use with tf.gather later
            self.rel_idxs = tf.constant(rel_idxs_np, dtype=tf.int32)
        elif self.strategy == "sig_5_points_full":
            # collect and concetenate the indexes of the relevant points for
            # both hands
            gathered_points = np.concatenate(
                (self.arch.sig_5_points_0_idxs,
                 self.arch.sig_5_points_1_idxs)
            )
            # add the additional relevant information to the relevant points
            rel_idxs_np = np.concatenate(
                (gathered_points,
                 self.arch.relevant_add_infos_idxs)
            )
            # convert them to a tensor for easy use with tf.gather later
            self.rel_idxs = tf.constant(rel_idxs_np, dtype=tf.int32)
        elif self.strategy == "sig_5_points_red":
            # collect and concetenate the indexes of the relevant points for
            # both hands
            rel_idxs_np = np.concatenate(
                (self.arch.sig_5_points_0_idxs,
                 self.arch.sig_5_points_1_idxs)
            )
            # convert them to a tensor for easy use with tf.gather later
            self.rel_idxs = tf.constant(rel_idxs_np, dtype=tf.int32)
        else:
            # set to false to signal that nothing for a slicing strategy got
            # initilised
            self.slicing_strategy_initialised = False

        # init base layer configs
        super().__init__(name=name, **kwargs)

    @tf.function
    def call(self, X):
        """heart of the layer: every batch will run through the graph that is
        created here

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        Returns:
            X (tf.Tensor): the batch, reduced according to the given strategy
                           shape: [batch_size, seq_len, STRATEGY_DEPENDENT]
                           It is also conceivable, that the seq_len-dimension
                           gets dropped here by a strategy (which is not yet
                           the case for any implemented strategy)
        """
        if self.debug_mode:
            tf.print("\n\n\n\n          in MAN_DIM_RED layer          ")
            tf.print("-- FOR MAXIMUM INFORMATION, DISSABLE --\n" \
                     "-- @tf.function DECORATOR AND ENABLE --\n" \
                     "--          EAGER EXECUTION          --\n")
            print("also eager in manual dimension reduce layer")

            tf.print("given STRATEGY: ", self.strategy)
            if self.slicing_strategy_initialised:
                tf.print("indexes for slicing strategy were initialised")
            else:
                tf.print("nothing for a slicing strategy was initialised")

            # this print will show a hugh output, but it will only print if the
            # @tf.function-decorator is uncommented and the model runs eagerly
            print("\ninput before reduction")
            print(X)

        # build the graphs, according to the given strategy
        if self.strategy == "_full":
            # the idxs to collect for this strategy are initialised in this
            # layers init, so the _simple_idx_reducer-func can easily be used
            X = self._simple_idx_reducer(X)
        elif self.strategy == "_red":
            # the idxs to collect for this strategy are initialised in this
            # layers init, so the _simple_idx_reducer-func can easily be used
            X = self._simple_idx_reducer(X)
        elif self.strategy == "del_z_full":
            # the idxs to collect for this strategy are initialised in this
            # layers init, so the _simple_idx_reducer-func can easily be used
            X = self._simple_idx_reducer(X)
        elif self.strategy == "del_z_red":
            # the idxs to collect for this strategy are initialised in this
            # layers init, so the _simple_idx_reducer-func can easily be used
            X = self._simple_idx_reducer(X)
        elif self.strategy == "avg_of_dims_full":
            X = self._avg_of_dims_full_reducer(X)
        elif self.strategy == "avg_of_dims_red":
            X = self._avg_of_dims_red_reducer(X)
        elif self.strategy == "sig_5_points_full":
            X = self._simple_idx_reducer(X)
        elif self.strategy == "sig_5_points_red":
            X = self._simple_idx_reducer(X)
        else:
            # signal if no strategy got triggered, possibly due to a wrongly
            # named strategy
            tf.print("\nWARNING! (Manual_Dim_Reducer)\nNo strategy got " \
                     "triggered by given keyword.\nBatch is passed without " \
                     "reduction")

        if self.debug_mode:
            tf.print("\nbatch shape after reduction: \n", X.shape)

            # this print will show a hugh output, but it will only print if the
            # @tf.function-decorator is uncommented and the model runs eagerly
            print("\noutput after reduction")
            print(X)
        return (X)

    def _simple_idx_reducer(self, X):
        """small function for simple reduction approaches, that just gather
        certain points. the indexes that shall be gathered are assigned in the
        init function of this layer, depending on the strategy

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]

        Returns:
            X (tf.Tensor): the given batched, reduced by the indexes that were
                           initialised in the __inti__, depending on the choosen
                           strategy
                           shape: [batch_size, seq_len, STRATEGY_DEPENDENT]
        """
        if self.debug_mode:
            tf.print("\nslice down to the indexes:\n", self.rel_idxs)
            tf.print("num indexes = ", self.rel_idxs.shape)
        # gather the relavant indexes, return reduced batch of instances
        X = tf.gather(X, self.rel_idxs, axis=2)
        return (X)

    def _average_of_dims(self, X):
        """helper function as this operation is needed for a _full and a _red
        graph, so this is centralized here. average slices per hand and
        dimension, return the concetenated results

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]

        Returns:
            hand_avgs (tf.Tensor): a given batch, reduced per hand and
                                   dimension. Therefore each frame is now
                                   represented by 6 values: 2handsx3dimensions
                                   shape: [batch_size, seq_len, 6]
        """
        # get the reduces means per dimension and hand, across the whole batch
        # hand_0
        hand_0_x_avg = self._mean_reduce_hlp(X, self.arch.hand_0_x)
        hand_0_y_avg = self._mean_reduce_hlp(X, self.arch.hand_0_y)
        hand_0_z_avg = self._mean_reduce_hlp(X, self.arch.hand_0_z)
        # hand_1
        hand_1_x_avg = self._mean_reduce_hlp(X, self.arch.hand_1_x)
        hand_1_y_avg = self._mean_reduce_hlp(X, self.arch.hand_1_y)
        hand_1_z_avg = self._mean_reduce_hlp(X, self.arch.hand_1_z)

        # concetenate the averages to a new batch, that represents the original
        # one by averaging the coordinates per hand and dimension, sothat one
        # frame representation will now have 6 values: one 3D point for hand_0
        # and one 3D point for hand_1
        hand_avgs = tf.concat(
            [hand_0_x_avg, hand_0_y_avg, hand_0_z_avg,
             hand_1_x_avg, hand_1_y_avg, hand_1_z_avg],
            2
        )

        if self.debug_mode:
            tf.print("\nreduce frame representations to means per hand and" \
                     " dimension")
            tf.print("\nhand-averages are: \n", hand_avgs)
        return (hand_avgs)

    def _avg_of_dims_full_reducer(self, X):
        """reduce all dimensions per hand to an average value, sothat the
        resulting tensor will contain one 3D point (-> three values) per hand,
        as well as additional informations like handedness or whatever,
        depending on the csv-architecture

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]

        Returns:
            X (tf.Tensor): the reduced batch. the new sequence len (NEW_REP_LEN)
                           depends on how many relevant informations are
                           contained and marked as relevant in the given
                           csv-architecture. NEW_REP_LEN will be
                           6+add_rel_infos
                           shape: [batch_size, seq_len, NEW_REP_LEN]
        """
        # get the averages per dimension and hand, across the whole batch
        hand_avgs = self._average_of_dims(X)
        # collect the additional relevant informations (might contain e.g.
        # handedness information etc.)
        additional_rel_infos = tf.gather(
            X,
            tf.constant(self.arch.relevant_add_infos_idxs),
            axis=2
        )
        # concetanate the averages coordinates with the additional informations
        # to the new batch
        X = tf.concat(
            [hand_avgs, additional_rel_infos],
            2
        )

        if self.debug_mode:
            tf.print("\nADDED additional information to the per-dim-and-\n" \
                     "hand reduced frame representation")
        return (X)

    def _avg_of_dims_red_reducer(self, X):
        """reduce all dimensions per hand to an average value, sothat the
        resulting tensor will contain on 3D point (-> three values) per hand

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]

        Returns:
            X (tf.Tensor): a given batch, reduced per hand and
                           dimension. Therefore each frame is now
                           represented by 6 values: 2handsx3dimensions
                           shape: [batch_size, seq_len, 6]
        """
        # get the averages per dimension and hand, across the whole batch
        X = self._average_of_dims(X)

        if self.debug_mode:
            tf.print("\nadded NONE additional information to the per-dim-\n" \
                     "and-hand reduced frame representation")
        return (X)

    def _mean_reduce_hlp(self, X, idxs):
        """small wrapper to return a reduced mean of the given indexes, per
        instance, over the batch, along the frame-representation-axis

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
            idxs (np.array): the indexes of the values to calculate the mean
                             over

        Returns:
            reduced_means (tf.Tensor): the means of the given indexes, across
                                       the whole batch
                                       shape: [batch_size, seq_len, 1]
        """
        reduced_means = tf.math.reduce_mean(
            tf.gather(X, idxs, axis=2),
            axis=2,
            keepdims=True
        )
        return (reduced_means)

    def compute_output_shape(self, input_shape):
        """this function is needed to seemingly use the custom layer

        Args:
            input_shape
        Returns:
            input_shape
        """
        return (input_shape)

    def get_config(self):
        """this function is needed to seemingly use the custom layer

        Returns:
            _configs (dict): an assignment of all variables that were used to
                             initialise the layer
        """
        config = {
            "strategy": self.strategy,
            "csv_struct": self.csv_struct,
            "debug_mode": self.debug_mode
        }
        base_config = super().get_config()
        return (dict(list(base_config.items()) + list(config.items())))


# decorator ensures easy loading of the custom layers
@keras.utils.register_keras_serializable(package="Custom", name="Passer")
class Passer(keras.layers.Layer):
    """use this empty layer to insert it during developement at any point in
    the model to see how the tensor look like, test operations, or do whatever
    """

    def __init__(self, print_arg="dummy", name="Passer", **kwargs):
        """initialise everything needed for this layer

        Args:
            print_arg (str): use this string if something specific shall be
                             printed by Passer-Layers, placed at different
                             points in the model
            name (str): name of the layer. defaults to tha class name. needs
                        to be specified if the layer is used multiple times, to
                        maintain uniqueness. Otherwise the API will throw an
                        error
        """
        self.print_arg = print_arg
        # init base layer configs
        super().__init__(name=name, **kwargs)

    def call(self, X):
        """heart of the layer: use this for whatever

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        Returns:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        """
        # play around here
        
        print("\nin passer")
        print(self.print_arg)
        print(X)
        print(X.shape)
        print("out passer")
        
        return (X)

    def compute_output_shape(self, input_shape):
        """this function is needed to seemingly use the custom layer

        Args:
            input_shape
        Returns:
            input_shape
        """
        return (input_shape)

    def get_config(self):
        """this function is needed to seemingly use the custom layer

        Returns:
            _configs (dict): an assignment of all variables that were used to
                             initialise the layer
        """
        config = {
            "print_arg": self.print_arg,
        }
        base_config = super().get_config()
        return (dict(list(base_config.items()) + list(config.items())))


if __name__ == "__main__":
    pass
