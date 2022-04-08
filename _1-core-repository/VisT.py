#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:35:40 2021

@author: marlinberger

This is the visualizer routine, that will analyse given video data with a
given tensorflow model.

Therefore, a folder need to be created in the "05_Evaluate" directory, whichs
name is detemindes beneath with the variable "EVAL_FOLDER". This folder
contains the folders:
    - 01_Processed_Videos
    - 02_Label_Tables
    - 03_Model

Inside the folder are:
01_Processed_Videos
    One folder for each video to be analysed. In this folder is not the mp4
    video but its extracted frames. This ensures there are no mistakes, as the
    same pictures are used as they were for the labeling process
02_Label_Tables
    The corresponding label tables for each video (-> folder) in
    01_Processed_Videos
03_Model
    The folder that is created for training runs under
    "99_Architectures/_saves". It contains a h5-model and a config file

Created output videos are layed imidiatly into the current EVAL_FOLDER-folder


"""
# python packages
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

# own packages
from motion_tracking_helpers import motion_tracking_helpers as mt_hlp
# this import is necessary sothat the custom_layers of the loaded model can be
# decoded
from motion_tracking_helpers import custom_layers

# User Inputs
# ------------
global EVAL_FOLDER, SCRIPT_MODE, LOCAL_BATCH_SIZE

# which evaluation should be done
EVAL_FOLDER = "TestEval_small"

# local batch size (use to avoid RAM overflow - how many instances are collected
# before a model's prediction is mixed in and the cache is cleared again)
LOCAL_BATCH_SIZE = 32
# ------------


def create_output_video(videopath, pic_paths_sorted, mp_results, predictions,
                        labels, Eval_Specs):
    """create a lit output video, that contains all informations that are
    intresting to investigate a model

    Args:
        videopath (PosixPath): the location of the used frames
        pic_paths_sorted (list): paths to all the frames in the videopath
        mp_results (mediapipe): the mediapipe results, easy to write on image
        predictions (np.array): predictions made for these frames
        labels (np.array): the labels, or just nan's, if no labels are given
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder
    """
    # create the video safe path
    video_safe_path = Path(
        f"{Eval_Specs.CURR_EVAL_MAIN_PATH}/" \
        f"{videopath.name}_analysed.avi"
    )
    # load an example image sothat the OutputWriter gets some insights on
    # the picture format
    example_pic = mt_hlp.load_tiff_image(pic_paths_sorted[0])
    # initialise the video output writer
    OutputWriter = mt_hlp.get_video_writer(
        video_safe_path, Eval_Specs, example_pic
    )

    # make some mediapipe assignments for easy access
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # initialise some custom draw settings
    draw_set = mt_hlp.get_cus_draw_settings()

    print("\nCreating the output video")
    # itterate through pictures
    for pic_path, mp_result, prediction, label in tqdm(zip(
            pic_paths_sorted, mp_results, predictions, labels),
            total=len(pic_paths_sorted)):
        # load the next picture
        pic = mt_hlp.load_tiff_image(pic_path)
        # Set flag to true
        pic.flags.writeable = True
        # Flip on horizontal to allign with mp output
        pic = cv2.flip(pic, 1)

        # storage for the wrist coordinates of the hands detected. Used to draw
        # the hand assignments
        num_coo_save = []
        # storage for the handedness results
        handedness_sto = []

        # rendering landmarks on the image
        if mp_result[1].multi_hand_landmarks:
            # itterate over all hands detected
            for num, (hand, handedness) in enumerate(zip(
                    mp_result[1].multi_hand_landmarks,
                    mp_result[1].multi_handedness)):
                # extract the handedness results
                handedness_clf = handedness.classification[0].label.lower()
                handedness_conf = handedness.classification[0].score
                # save them
                handedness_sto.append((handedness_clf, handedness_conf))

                # get and save the wrist coordinates for later assignment
                wrist_coordinates_n = (
                    int(hand.landmark[0].x * pic.shape[1]),
                    int(hand.landmark[0].y * pic.shape[0])
                )
                num_coo_save.append(wrist_coordinates_n)

                # get color dependend on handedness prediction
                hand_color = draw_set[handedness_clf]
                # draw the hand skeleton
                mp_drawing.draw_landmarks(
                    pic, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=draw_set["outline"],
                                           thickness=1, circle_radius=4),
                    mp_drawing.DrawingSpec(color=hand_color,
                                           thickness=2, circle_radius=2),
                )

        # Flip back horizontal
        pic = cv2.flip(pic, 1)
        # Make into PIL Image as this is way nicer to handle for writing on
        # it. Espacially monospace-fonts are available
        pic_PIL = Image.fromarray(pic)
        # Get a drawing context
        draw = ImageDraw.Draw(pic_PIL)

        # tag the plotted coordinates with the vector assignment's number
        for num, coo in enumerate(num_coo_save):
            # correct the coordinates as they are mirrored
            org = (
                pic.shape[1] - coo[0] - draw_set["vec_num_corr"],
                coo[1] - 2 * draw_set["vec_num_corr"]
            )
            # write the text
            txt = str(num)
            draw.text(
                org,
                txt,
                draw_set["vec_num_color"],
                font=draw_set["vec_num_font"])

        # get strings that are to draw on the frame
        (handedness_results_print, models_predictions_print, ground_truth_print,
         info_tag_print) = (
            mt_hlp.get_eval_text(handedness_sto, prediction, label, Eval_Specs)
        )

        # write the models predictions (inf1)
        draw.text(
            draw_set["inf1_pred_pos"],
            handedness_results_print,
            draw_set["inf1_pred_color"],
            font=draw_set["inf1_pred_font"])

        # write the models predictions (inf2)
        draw.text(
            draw_set["inf2_pred_pos"],
            models_predictions_print,
            draw_set["inf2_pred_color"],
            font=draw_set["inf2_pred_font"])

        # draw ground truth string
        draw.text(
            draw_set["ground_truth_pos"],
            ground_truth_print,
            draw_set["ground_truth_color"],
            font=draw_set["ground_truth_font"])

        # draw base info string
        draw.text(
            draw_set["base_info_pos"],
            info_tag_print,
            draw_set["base_info_color"],
            font=draw_set["base_info_font"])

        # Convert back to OpenCV image
        pic = np.array(pic_PIL)

        # write the frame
        OutputWriter.write(pic)
        cv2.imshow("frame", pic)

    # release and destroy everything when the job is done
    OutputWriter.release()
    cv2.destroyAllWindows()


def video_overlay_outline():
    """this function forms the foundation to create the video overlays
    """
    # itterate over all input videos
    print("\nItterating over videos")
    for videopath in tqdm(Eval_Specs.VIDEO_PATHS, disable=False):
        # initialise flag if labels exist or not
        labled_data = False

        # get all picture paths, sort them by framenumber
        pic_paths = mt_hlp.get_filepaths_in_dir(videopath)
        pic_paths_sorted = mt_hlp.sort_paths_by_first_num(pic_paths)

        # create the corresponding label vector if a label table exists
        label_tab_path = [path for path in Eval_Specs.LABEL_TABLE_PATHS
                          if videopath.name == str(path.name).split(".")[0]]

        # NOTE: this should be error checked on the case that more than
        # 		one path occures
        if len(label_tab_path) == 1:
            # get the motion-class vector from the label table, if one is
            # given
            motionclass_vector = mt_hlp.get_motionclass_vector(
                label_tab_path[0], len(pic_paths_sorted)
            )
            labled_data = True
        elif len(label_tab_path) == 0:
            # initialise a list of nan's if no labels are given
            placeholder_array = np.empty((len(pic_paths_sorted)))
            placeholder_array[:] = np.NaN
            motionclass_vector = list(placeholder_array)

        # reduce frames if choosen
        red_fac = Eval_Specs.FPS_REDUCTION_FAC
        pic_paths_sorted = pic_paths_sorted[::red_fac]
        motionclass_vector = motionclass_vector[::red_fac]

        # collect the mediapipe results, to ensure both: not to much space
        # is needed and also every picture is only looked up once.
        # this results array should take around 300 MB space for a train
        # video (which would be about 15m long (yes?!))
        print("\nProcess frames with mediapipe (inf1)")
        mp_results = []
        for framenum, (pic_path, motionclass) in enumerate(tqdm(
                zip(pic_paths_sorted, motionclass_vector),
                total=len(pic_paths_sorted),
                leave=False)):
            # get the numpy representation of the picture
            image = mt_hlp.load_tiff_image(pic_path)
            # process the image
            results = mt_hlp.mediapipe_hands_process(image, paramset)
            mp_results.append((motionclass, results))

        # import needs to be behind the call of mediapipe, as they rely on
        # tf-lite and the programm will throw a conflict otherwise. This
        # should probably get solved in future tf versions
        import tensorflow as tf

        # stack coordinates, create the instances along with the labels
        predictions = []
        labels = []
        # this storage is evaluated and cleared every LOCAL_BATCH_SIZE to
        # avoid RAM overflow due to the stacked instances
        instances_temp = []
        # count steps to trigger the model every LOCAL_BATCH_SIZE-time
        cnt = 0
        print("\nGet predictions from skeleton model (inf2)")
        # preprocess the range-list for easy use of tqdm
        for idx in tqdm(range(Eval_Specs.SEQ_LEN, len(pic_paths_sorted) + 1)):
            # get the in- and out points per instance
            idx_0, idx_1 = idx - Eval_Specs.SEQ_LEN, idx
            # get the label
            label = motionclass_vector[idx_1 - 1]

            # extract coordinates per instance
            buffer = mp_results[idx_0:idx_1]
            # get the label from the last picture
            label = buffer[-1][0]
            # convert the mediapipe results to the csv format used
            buffer_conv = [mt_hlp.convert_mediapipe_output_v2(inst[1]) for
                           inst in buffer]
            # easaly stack them with the little turn over pandas
            temp_df = pd.DataFrame.from_dict(buffer_conv)
            temp_df = temp_df.values

            # safe the instance and it's label
            instances_temp.append(temp_df)
            labels.append(label)

            # process local batches
            cnt += 1
            if cnt == LOCAL_BATCH_SIZE:
                # convert instances to tensor of shape [batch, seq., coo.]
                temp_batch = tf.convert_to_tensor(
                    np.array(instances_temp), dtype=tf.float32
                )
                # predict the temp batch
                predictions_temp = Eval_Specs.model.predict(temp_batch)
                # save the prediction
                predictions.extend(predictions_temp)
                # reset the temp-batch-stuff
                instances_temp = []
                cnt = 0

        # predict the last instances that were not covered (if there are
        # any)
        # NOTE: little redundance, but not worth the function
        if len(instances_temp) != 0:
            temp_batch = tf.convert_to_tensor(
                np.array(instances_temp), dtype=tf.float32
            )
            # predict the temp batch
            predictions_temp = Eval_Specs.model.predict(temp_batch)
            # save the prediction
            predictions.extend(predictions_temp)

        # add dummy values for the pictures that are not predicted in the
        # beginning, due to the sequential input
        # get difference between prediction and pictures
        diff = len(mp_results) - len(predictions)
        # create nan-prediction arrays for these first pictures
        dummy_pred = np.empty((len(predictions[-1])))
        dummy_pred[:] = np.NaN
        dummy_pred_stack = [dummy_pred for _ in range(diff)]
        # put them in front of the predictions
        dummy_pred_stack.extend(predictions)
        predictions = dummy_pred_stack
        # do the same for the labels. they have one dimension less, so
        # a slightly simpler approach can be taken
        dummy_labels = [np.NaN for _ in range(diff)]
        dummy_labels.extend(labels)
        labels = dummy_labels

        # convert the prediction and labels to np.array, for fast
        # vectorized calculus and easy handling as nothing more needs to be
        # appended
        predictions = np.array(predictions)
        labels = np.array(labels)

        # pass everything to the routine, that created a nitty gritty
        # video output
        create_output_video(
            videopath,
            pic_paths_sorted,
            mp_results,
            predictions,
            labels,
            Eval_Specs
        )


def error_checks(Eval_Specs):
    """function to ensure errors are detected and printed

    Args:
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder

    Returns:
        all_good (bool): indicate if everythings all right
        error_str (str): used to collect all failed safty check and print them
                         to the person using the script. this way, all errors
                         can be signaled at once
    """
    # initialise the return param
    all_good = True
    error_str = ""

    """
    TODO: include following checks
        - only one model w. _config file exists in the models folder as 
          subfolder with the run's name

    """

    # if the error_str contains anything, set all_good-flag to false to not
    # start the train data creation. add intro line to error_str
    if len(error_str) != 0:
        all_good = False
        error_str = mt_hlp.adjust_error_message(error_str)

    return (all_good, error_str)


if __name__ == "__main__":

    # collect all information about the choosen evaluation folder set
    Eval_Specs = mt_hlp.Eval_Specs(EVAL_FOLDER, MODE="VisT")
    # get the paramset, with which all pictures shall be evaluated by mediapipe
    # NOTE: CHANGING THIS IS THE DANGER ZONE!
    paramset = mt_hlp.mediapipe_hands_paramsets(0)

    # perform error checks
    all_good, error_str = error_checks(Eval_Specs)

    # start the evaluation if everything is ok, or else print whats wrong
    if all_good:
        video_overlay_outline()
    else:
        # print whats wrong if so
        print(error_str)
