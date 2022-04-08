#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:03:40 2021

@author: marlinberger

helper functions in the context of the training of inference 2 (motion
classification)
"""

# python packages
import csv
import cv2
import logging
import mediapipe
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from PIL import ImageFont
from tensorflow import keras



def check_import():
    """use this function to check if the import worked
    """
    logging.debug("\nyou made it in\n")


def initialise_logger(depth="quiet"):
    """initialise a logger
    """
    # keep most logs down
    if depth == "quiet":
        logging.basicConfig(level=logging.ERROR)
    # good go-to setting
    if depth == "smooth":
        logging.basicConfig(level=logging.WARNING)
    # get a bit more information
    if depth == "info":
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S'
        )



def name_constants(keyword):
    """returns the corresponding file- or foldername to a keyword. This way,
    they are defined at a fixed place, but are accessable all over the project

    Args:
        keyword (str): used to access the corresponding name

    Returns:
        name (str): (file-)name of anything
    """
    # folders
    # --------
    # names of the folders used to shift and create the train-ready data
    RAW_VIDEO_DIR_NAME = "00_Raw_Inputs"
    RENAMED_INPUTS_DIR_NAME = "01_Renamed_Inputs"
    LABEL_TABBLE_DIR_NAME = "02_Label_Tables"
    TRAIN_DATA_DIR_NAME = "03_Train_Data"
    TRAIN_DATA_PACKS_DIR_NAME = "04_Train_Data_Packs"
    EVALUATE_DIR_NAME = "05_Evaluate"
    ANALYZE_DIR_NAME = "06_Analyze"
    ARCHITECTURE_DIR_NAME = "99_Architectures"
    # save folder for models
    MODEL_SAVE_DIR_NAME = "_saves"
    # subfolder in "05_Evaluate"
    EV_BASE = "01_Processed_Videos"
    EV_LABELS = "02_Label_Tables"
    EV_MODEL = "03_Model"
    EV_RESULTS = "04_Results"
    # subfolder in "06_Analyze" for the plot saves
    PLOT_SAVES = "_plot-results"
    PICKLED_DEV_BLOCKS = "_pickled_dev_blocks"
    # subfolder in "06_Analyze" per Analysation folder
    AN_MODEL = "01_Models"

    # files
    # --------
    # name of the _config file for neural nets
    CONFIG_FILE_NAME = "_config.csv"

    # suffix's
    # --------
    # Ending for the table-frame-motionclass tables
    TABLES_SUFFIX = ".tab.txt"

    # serach through just created local variables by keyword
    name = locals()[keyword]
    return (name)


def assignments(keyword):
    """keep assignments/maskings for strings here, on a central place, to
    have them in one place and be able to later look them up easaly

    Args:
        keyword (str): the keyword to maks/encode

    Returns:
        masked_keyword (int): mask for the keyword given
    """
    # for the hand assignments
    if keyword == "left":
        return (0)
    if keyword == "right":
        return (1)


def get_filepaths_in_dir(path_to_dir, ignore_hidden_files=True):
    """return all paths from files inside the given directory

    Args:
        path_to_dir (PosixPath): path to the dir, from which the folders shall
                                 be returned
        ignore_hidden_files (bool): wether or not to include hidden files

    Returns:
        paths (list(PosixPath)): paths to the files inside the given directory
    """
    if ignore_hidden_files:
        paths = [x for x in path_to_dir.iterdir() if not x.name[0] == "."]
    else:
        paths = [x for x in path_to_dir.iterdir()]
    return (paths)


def get_unique_path_variation(check_path, exclude_suffix=True):
    """function checks a desired path, that shall be created. If it already
    exists, the function will append an underline and a number - starting at 0.
    It then re-checks if this path already exists and will continue to count
    up and append numbers, until it found a valid variation.
    The found path is returned to be used for the file creation.
    NOTE: simply checking if a file exits is not suffiecient if the extension
          is not known
    
    Args:
        check_path (PosixPath): path to validate and check
        exclude_suffix (bool): wether e.g. a ".png" shall also be taken in
                               account or not
    
    Returns:
        valid_path (PosixPath): a variation of the given path, that can safely
                                be created
    """
    # get all paths in the target dir
    all_paths = get_filepaths_in_dir(check_path.parent)
    # possibly cut away theire suffix
    if exclude_suffix:
        check_name = str(check_path.stem)
        all_filenames = [str(path.stem) for path in all_paths]
    else:
        check_name = str(check_path.name)
        all_filenames = [str(path.name) for path in all_paths]
    
    # check if there are already files with the same name
    doubled_names = list(filter(lambda x: str(x) == check_name, all_filenames))
    
    # if there are doubles, append a number to the name and check if this
    # name already exists. Count the number up until the name is unique
    if len(doubled_names) > 0:
        num2app = 0
        while True:
            try_name = f"{check_name}_{num2app}"
            if try_name in all_filenames:
                num2app += 1
            else:
                valid_name = try_name
                break
        # create the valid path
        valid_path = check_path.parent / (valid_name + str(check_path.suffix))
    else:
        valid_path = check_path
    
    return(valid_path)


def get_folderpaths_in_dir(path_to_dir):
    """return all paths from folders inside the given directory

    Args:
        path_to_dir (PosixPath): path to the dir, from which the folders shall
                                 be returned

    Returns:
        paths (list(PosixPath)): paths to the folders inside the given directory
    """
    paths = [x for x in path_to_dir.iterdir() if x.is_dir()]
    return (paths)


def mediapipe_hands_paramsets(paramset_n):
    """save packs of parameters, with which the mediapipe_hands solution
    is driben, here. This way, different versions can easaly be developed and
    used across several devices and architectures

    Args:
        paramset_n (int): index of the param set to return

    Return:
        paramset (dict): params to use with the mediapip-hands solution
    """

    # initial params to use
    if paramset_n == 0:
        paramset = {
            "static_image_mode": True,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        }

    # NOTE: if "max_num_hands" is set to something else than 2, it will break
    # 		the current structure of how the mediapipe results are saved and
    # 		send across the stack (this applies to convert_mediapipe_output_v2()
    # 		function. convert_mediapipe_output() can handle anything but needs
    # 		different input pipelines for the tensorflow models)
    paramset["max_num_hands"] = 2

    return (paramset)


def load_tiff_image(pic_path):
    """load a tiff picture with the give path

    Args:
        pic_path (PosixPath): the path to the picture

    Returns:
        image (np.ndarray): the loaded image
    """
    # this is the way to get tiff pics in
    image = cv2.imread(str(pic_path), -1)
    return (image)


def mediapipe_hands_process(image, paramset):
    """routine to process straigt images (NOT recorded by a selfie cam) with
    the mediapipes hands solution.

    Args:
        image (numpy.ndarray): the picture, probably loaded with cv2, at least
                               in the training context
        paramset (dict): the params to be used, bundled in a dict. For shape,
                         look at the function of this script:
                         mediapipe_hands_paramsets()

    Returns:
        results (mediapipe.[.].SolutionOutputs): The results from mediapipe
                                                 hands
    """
    with mediapipe.solutions.hands.Hands(
        static_image_mode=paramset["static_image_mode"],
        max_num_hands=paramset["max_num_hands"],
        min_detection_confidence=paramset["min_detection_confidence"],
        min_tracking_confidence=paramset["min_tracking_confidence"],
    ) as hands:
        # BGR 2 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # flip on horizontal, as mediapipe expects the output from a
        # front camera (this way, "handedness" works smoothly)
        image = cv2.flip(image, 1)
        # set flag
        image.flags.writeable = False
        # Get the mediapipe output
        results = hands.process(image)
    return (results)


def convert_mediapipe_output(mediapipe_result, ts_ms=0, ts_img=0):
    """convert the mediapipe result to the shape in which it will be sent across
    the mqtt stack. It is also used in the context of creating the training
    data, to ensure all formats are exactly the same.
    The structure is designed to be extantable without breaking older structures
    and looks like this:
        structured_output = {"multi_hand_landmarks":
                                {"hand_1": {"0": [0.304534, 0.5123, 0.92767],
                                            "1": [0.73456, 0.12542, 0.7352]
                                            },
                                 "hand_2": {"0": [0.304534, 0.5123, 0.92767],
                                            "1": [0.73456, 0.12542, 0.7352]
                                            }
                                },
                            "multi_handedness":
                                {"hand_1": {"hand_pred": "left",
                                            "hand_pred_conf": 0.53
                                            },
                                {"hand_2": {"hand_pred": "right",
                                            "hand_pred_conf": 0.88
                                            }
                                }
                            }

    Args:
        mediapipe_result (mediapipe.process): the output from the mediapipe
                                              analysis
        ts_ms (int): timestamp [datetime], of when the photo command was given
        ts_img (int): timestamp [datetime], of when the photo was taken

    Returns:
        structured_output (dict): the structures output
    """
    # extract coordinates and mediapipe's hand prediction
    coordinates = mediapipe_result.multi_hand_landmarks
    hands = mediapipe_result.multi_handedness

    # initialise the hand-representation storage that will be passed along the
    # pipeline in every aspect
    structured_output = dict()

    # apply timestamp
    structured_output["timestamp_photo_command"] = ts_ms
    structured_output["timestamp_image_taken"] = ts_img
    # initialise further structre
    structured_output["multi_hand_landmarks"] = dict()
    structured_output["multi_handedness"] = dict()
    structured_output["no_hands_detectec_flag"] = False

    # write the results for detected hands
    if isinstance(coordinates, list):
        for idx, landmarks in enumerate(coordinates):
            # under this name, all informations for this hand are saved
            handname = f"hand_{idx}"

            # assign the mediapipe hands results
            # initialise the landmarks sto
            structured_output["multi_hand_landmarks"][handname] = {}
            # shorten calls
            fill_hand = structured_output["multi_hand_landmarks"][handname]
            # fill in the coordinates for the hand
            for idx_k, coordinate in enumerate(landmarks.landmark):
                fill_hand[idx_k] = [coordinate.x,
                                    coordinate.y,
                                    coordinate.z]

            # assign the mediapipe handedness results
            label = hands[idx].classification[0].label.lower()
            score = hands[idx].classification[0].score
            structured_output["multi_handedness"][handname] = {
                "hand_pred": label,
                "hand_pred_conf": score
            }
    # catch the case that no hand is detected
    else:
        # signal that no hand was detected
        structured_output["no_hands_detected_flag"] = True

    return (structured_output)


def convert_mediapipe_output_v2(mediapipe_result, ts_ms=0, ts_img=0):
    """convert the mediapipe result to the shape in which it will be sent across
    the mqtt stack. It is also used in the context of creating the training
    data, to ensure all formats are exactly the same.
    The structure is designed to be extantable without breaking older structures
    and looks like this, but has a different shape than the v1-function. It is
    structured as a 1-line-table. This way, processing is SUPER efficient, while
    the readability suffers, BUT, these files wont be read anyway. And if they
    shall be read - during debugging etc. - they are still readable, just a
    little bit less nice
    This approach assumes that the "max_num_hands" argument for the mediapipe
    solution is set to 2.

    Args:
        mediapipe_result (mediapipe.process): the output from the mediapipe
                                              analysis
        ts_ms (int): timestamp [datetime], of when the photo command was given
        ts_img (int): timestamp [datetime], of when the photo was taken

    Returns:
        structured_output (dict): the structures output
    """
    # extract coordinates and mediapipe's hand prediction
    coordinates = mediapipe_result.multi_hand_landmarks
    hands = mediapipe_result.multi_handedness

    # initialise the hand-representation storage that will be passed along the
    # pipeline in every aspect
    structured_output = dict()

    # apply timestamp
    structured_output["timestamp_photo_command"] = ts_ms
    structured_output["timestamp_image_taken"] = ts_img
    structured_output["no_hands_detected_flag"] = 0

    # initialise hand-coordinate columns
    for hand_num in [0, 1]:
        # insert the mediapipes prediction wether it is a left or right hand
        # initialise both with -1, which signals no hand got detected (nan's
        # are thus exclusive to coordinates, which has some benefits for
        # imputation layers)
        structured_output[f"{hand_num}_pred"] = -1
        structured_output[f"{hand_num}_pred_conf"] = -1
        for hand_point in range(21):
            structured_output[f"{hand_num}_{hand_point}_x"] = np.nan
            structured_output[f"{hand_num}_{hand_point}_y"] = np.nan
            structured_output[f"{hand_num}_{hand_point}_z"] = np.nan

    # itterate through mediapipe results and assign them. if there are no, this
    # part is skipped and the handcoordinates are already filled withs "False"
    if isinstance(coordinates, list):
        for hand_num, landmarks in enumerate(coordinates):
            # assign the hand prediction. left == 0, right == 1
            hand_masked = assignments(
                hands[hand_num].classification[0].label.lower()
            )
            # safe the handedness results
            structured_output[f"{hand_num}_pred"] = hand_masked
            structured_output[f"{hand_num}_pred_conf"] = (
                hands[hand_num].classification[0].score)

            # fill up the hand points
            for hand_point, coordinate in enumerate(landmarks.landmark):
                structured_output[f"{hand_num}_{hand_point}_x"] = coordinate.x
                structured_output[f"{hand_num}_{hand_point}_y"] = coordinate.y
                structured_output[f"{hand_num}_{hand_point}_z"] = coordinate.z
    else:
        # signal that no hand was detected
        structured_output["no_hands_detected_flag"] = 1

    return (structured_output)


def write_coordinates_to_csv(filepath, mediapipe_result):
    """helper function, to save the mediapipe-output as csv file.

    Args:
        filepath (PosixPath): where to safe the file. includes the filename
        structured_output (dict): the processed output from the mediapipe
                                  analysis, in the structure that is used
                                  across the project stack (see function
                                  convert_mediapipe_output() above)
    """
    # open the given file, write it key by key on the first dictionary level
    with open(str(filepath) + ".csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mediapipe_result.items():
            writer.writerow([key, value])


def write_coordinates_to_tfrecord(filepath, mediapipe_result):
    """helper function, to save the mediapipe-output as tfrecord file. This
    will speed up the reading during training

    Args:
        filepath (PosixPath): where to safe the file. includes the filename
        structured_output (dict): the processed output from the mediapipe
                                  analysis, in the structure that is used
                                  across the project stack (see function
                                  convert_mediapipe_output() above)
    """
    # TODO: das hier machen falls es sich als bottleneck zeigt
    pass


def open_labels_table(tablepath):
    """open a labels table under the given path. Return the lines as a list

    Args:
        tablepath (PosixPath): path where to find the table

    Retuns:
        lines (list(str)): the lines in the table. each list entry represents
                           one line
    """
    with open(tablepath) as file:
        # read line by line
        lines = file.readlines()
        # strip backslashs etc.
        lines = [line.rstrip() for line in lines]
    return (lines)


def adjust_error_message(error_str):
    """routine to adjust the error message returned, during the safty checks
    of the train data creation

    Args:
        error_str (str): contains the error message(s)

    Returns:
        error_str_adjusted: the adjusted error string
    """
    error_str_adjusted = (
        "\n------\nERROR!\n------\n" + error_str + "\n------\n" \
        "Remove data inconsistencies, restart the script" \
        "\n------\n"
    )
    return (error_str_adjusted)


def get_abs_script_path(inter_file_var):
    """return the absolute path of a script. To do so, just pass the scripts
    intern __file__ variable

    Args:
        inter_file_var (str): the call-script intern __file__ file variable

    Returns:
        abs_path (PosixPath): absolute path to the script which called this
                              function
    """
    abs_path = Path(os.path.dirname(os.path.abspath(inter_file_var)))
    return (abs_path)


def df_from_model_config(configfile_path):
    """return a clean dataframe representation of a model's config file, which
    path is given to this function

    Args:
        configfile_path (PosixPath): path to the config file

    Returns:
        config_df (pd.Dataframe): The dataframe representation of the csv config
                                  file
    """
    # load the file
    config_df = pd.read_csv(configfile_path, header=None).transpose()
    # reallign indexes etc.
    config_df.reset_index(drop=True, inplace=True)
    config_df.columns = config_df.iloc[0]
    config_df.drop(index=0, inplace=True)
    return (config_df)


def sort_paths_by_first_num(paths):
    """sort given paths by a number, that comes first in the file's name
    and are delimited by an underline from the rest of the file's name

    Args:
        paths (list(PosixPath)): a list of paths

    Returns:
        sorted_paths (list(PosixPath)): the same list of paths, but sorted by
                                        number in the beginning
    """
    sorted_paths = sorted(paths, key=lambda x: int(x.name.split("_")[0]))
    return (sorted_paths)


def get_motionclass_vector(tablepath, tot_frame_num):
    """return an assignment for every frame with it's motion class, which
    can be used to loop through all picture and have theire motion class
    appearand

    Args:
        tablepath (PosixPath): path where to find the table
        tot_frame_num (int): how many frames the inspected video contains

    Returns:
        motionclass_vector (list): len equals the given input pictures, each
                                   each position contains the motion class for
                                   this frame
    """
    # open the label table
    table_lines = open_labels_table(tablepath)

    # open storage for the value pairs
    label_frame_assignment = []
    for value_pair in table_lines:
        # extract each startframe and motionclass
        startframe, motionclass = value_pair.split(",")
        # store them as bundled integers
        label_frame_assignment.append((int(startframe), int(motionclass)))

    # the label vector, containing one entry per picture
    motionclass_vector = []
    for idx, (startframe, motionclass) in enumerate(label_frame_assignment):
        # determine for how many pictures the motionclass appears.
        # distinguish the way this is determined for the last itteration
        if idx < len(label_frame_assignment) - 1:
            class_present_for = (label_frame_assignment[idx + 1][0] -
                                 startframe)
        else:
            class_present_for = tot_frame_num - startframe
        # build the motionclass vector by appending the motion class for
        # so many times, like frames they appear
        motionclass_vector.extend([motionclass for _ in range(
            class_present_for)])

    return (motionclass_vector)


def get_video_writer(video_safe_path, Eval_Specs, example_pic):
    """return an object that can be used to create a video under the given path

    Args:
        safepath (PosixPath): the path to the video that shall be created
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder
        example_pic (np.arr): a representative image for the video that's about
                              to be created

    Returns:
        OutputWriter (cv2 object): the initialised video writer
    """
    # calculate the output framerate NOTE: if train-fps differs from
    # evaluate-fps, this needs to be considered somewhere and probably the
    # calculation here needs to be performed differently
    fps = Eval_Specs.ORIGINAL_FPS / Eval_Specs.FPS_REDUCTION_FAC

    # set the output shape
    dims = example_pic.shape[1], example_pic.shape[0]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    OutputWriter = cv2.VideoWriter(str(video_safe_path), fourcc, fps, dims)
    return (OutputWriter)


def get_cus_draw_settings():
    """return some custom draw settings used to create the evaluation videos

    Returns:
        draw_set (dict): contains plot settings
    """

    draw_set = {
        # set colors to be used wether the hand is detected as right or left
        "right": (70, 152, 235),
        "left": (235, 205, 70),
        "outline": (50, 50, 50),
        # settings for the hand assignment at the wrist position
        "vec_num_font": ImageFont.truetype(
            "/Library/Fonts/Andale Mono.ttf",
            60
        ),
        "vec_num_color": (238, 0, 255),
        "vec_num_corr": 35,
        # settings for the model predictions (inf1 - mediapipe)
        "inf1_pred_pos": (950, 40),
        "inf1_pred_color": (255, 255, 255),
        "inf1_pred_font": ImageFont.truetype(
            "/Library/Fonts/Andale Mono.ttf",
            32
        ),
        # settings for the model predictions (inf2 - custom)
        "inf2_pred_pos": (30, 40),
        "inf2_pred_color": (255, 255, 255),
        "inf2_pred_font": ImageFont.truetype(
            "/Library/Fonts/Andale Mono.ttf",
            32
        ),
        "inf2_show_best_n": 3,
        # settings for the ground truth labe print
        "ground_truth_pos": (600, 40),
        "ground_truth_color": (255, 255, 255),
        "ground_truth_font": ImageFont.truetype(
            "/Library/Fonts/Andale Mono.ttf",
            32
        ),
        # settings for the base information print
        "base_info_pos": (30, 900),
        "base_info_color": (255, 255, 255),
        "base_info_font": ImageFont.truetype(
            "/Library/Fonts/Andale Mono.ttf",
            32
        ),
    }
    return (draw_set)


def get_eval_text(handedness_sto, prediction, label, Eval_Specs):
    """return the text that shall be layed over evaluated videos

    Returns:
        text (str): a text that can be put straigt on a video frame
    """
    # get some draw settings
    draw_set = get_cus_draw_settings()

    # process the handedness results into a string
    handed_print = [[np.nan, np.nan], [np.nan, np.nan]]
    for idx, hand_res in enumerate(handedness_sto):
        handed_print[idx][0] = hand_res[0]
        handed_print[idx][1] = hand_res[1] * 100
    handedness_results_print = (
        f"Hand predictions:\n" \
        f" 0: {handed_print[0][0]:<5}, {handed_print[0][1]:5.1f} %\n" \
        f" 1: {handed_print[1][0]:<5}, {handed_print[1][1]:5.1f} %"
    )

    # process the models prediction into a string
    pred_pre = []
    # distinguish between if there are any predictions or not. if there are non,
    # this will be indicated by the predictions only being nan's, in which case
    # the else block is triggered
    if not np.isnan(np.sum(prediction)):
        highest_probs = np.flip(np.argsort(prediction))
        for pred_n in range(draw_set["inf2_show_best_n"]):
            pred_pre.append(
                (highest_probs[pred_n],
                 prediction[highest_probs[pred_n]])
            )
    else:
        pred_pre = [("-", np.nan) for _ in range(draw_set["inf2_show_best_n"])]
    # format the output
    models_predictions_print = (
        f"Top {draw_set['inf2_show_best_n']} motion classes:\n" \
        f" 1: class {pred_pre[0][0]:<1}  conf {pred_pre[0][1] * 100:4.1f} %\n" \
        f" 2: class {pred_pre[1][0]:<1}  conf {pred_pre[1][1] * 100:4.1f} %\n" \
        f" 3: class {pred_pre[2][0]:<1}  conf {pred_pre[2][1] * 100:4.1f} %"
    )

    # section to process and compare prediction to ground truth (if given)
    ground_truth_print = (
        f"Ground truth:\n" \
        f" class: {label}\n" \
        f" eval : {label == pred_pre[0][0]}"
    )

    # tag from the script used for training and the run's name
    info_tag_print = (
        f"Base: {Eval_Specs.model_config['Script']}\n" \
        f"Run : {Eval_Specs.RUN_NAME}\n" \
        f"FPS : {Eval_Specs.ORIGINAL_FPS / Eval_Specs.FPS_REDUCTION_FAC:.2f}\n"\
        f"Seq : {Eval_Specs.SEQ_LEN}\n" \
        )

    return (handedness_results_print, models_predictions_print,
            ground_truth_print, info_tag_print)


class Eval_Specs():
    """an object of this class will contain information about the folder
    structure, model used etc., which are needed at different places and are
    therefore commulated inside this object
    """

    def __init__(self, MAIN_FOLDER, MODE):
        """initialise the eval-spec object

        Args:
            MAIN_FOLDER (str): the name of the evaluation folder to be evaluated
            mode (str): in ["VisT", "Analyzer"]. Indicates what's done
        """
        # get paths and names
        self.MAIN_FOLDER = MAIN_FOLDER
        self.MODE = MODE
        self.CURR_EVAL_MAIN_PATH = self.get_curr_eval_path()

        # get the names and paths of all models
        self.TRAIN_RUN_NAMES, self.TRAIN_RUN_PATHS = self.get_train_run_infos()

        # as for the VisT-Routine only one model is appearent, everything can
        # be initialised imidiatly
        if self.MODE == "VisT":
            # initialise video specific stuff
            self.PATH_TO_VIDEOS_MAIN = Path(
                f"{self.CURR_EVAL_MAIN_PATH}/" \
                f"{name_constants('EV_BASE')}"
            )
            self.VIDEO_PATHS = get_folderpaths_in_dir(
                self.PATH_TO_VIDEOS_MAIN
            )
            self.INPUT_VIDEOS_N = len(self.VIDEO_PATHS)
            # get the label tables, which is also only needed for the video-
            # overlay-routine
            self.LABEL_TABLE_PATHS = get_filepaths_in_dir(
                self.CURR_EVAL_MAIN_PATH / name_constants("EV_LABELS")
            )
            # intialise the model specific stuff
            self.setup_on_run(self.TRAIN_RUN_PATHS[0])
        
        elif self.MODE == "Analyzer":
            # if the Analyzer routine uses this object, it will re-initialize
            # this object on evey run/model on it's own, by calling the object's
            # function self.setup_on_run() from a loop that itterates through
            # all detected model paths
            pass

        # set up a central saving instance, that can be used for many things.
        # it can be updated with the function self.add_information(...), see
        # function description for further information
        self.eval_storrage = {}

        # set up potential save flag and path
        self.save_figures = False
        self.save_base_path = False
        # set up potential storage for fetched wandb run data
        self.fetched_run_data_available = False
        self.fetched_run_data = False

    def get_curr_eval_path(self):
        """return the path to the folder from which everything starts

        Returns:
            CURR_EVAL_MAIN_PATH (PosixPath): path to the current eval dir
        """
        # get the name of the folder in which the outer eval/analyze-folders lie
        if self.MODE == "VisT":
            TOPLEVEL_FOLDER = name_constants("EVALUATE_DIR_NAME")
        elif self.MODE == "Analyzer":
            TOPLEVEL_FOLDER = name_constants("ANALYZE_DIR_NAME")
        
        # get the path of the main eval/analyzation folder
        CURR_EVAL_MAIN_PATH = Path(
            f"{TOPLEVEL_FOLDER}/" \
            f"{self.MAIN_FOLDER}"
        )

        return (CURR_EVAL_MAIN_PATH)

    def get_train_run_infos(self):
        """extract the name of the run, whichs model got placed in the choosen
        evaluation folder to be used

        Returns:
            TRAIN_RUN_NAMES (list(str)): the names of the runs from the models
                                         used
            TRAIN_RUN_PATHS list((PosixPath)): paths to the folders of the
                                               models used
        """
        # get the name of the folder in which models are to be found
        if self.MODE == "VisT":
            MODEL_SUB_PATH = name_constants("EV_MODEL")
        elif self.MODE == "Analyzer":
            MODEL_SUB_PATH = name_constants("AN_MODEL")
        
        # get the path where the models are layed
        MODEL_FOLDER_PATH = Path(
            f"{self.CURR_EVAL_MAIN_PATH}/" \
            f"{MODEL_SUB_PATH}"
        )

        # get the paths and names of the models (== runs) in there
        TRAIN_RUN_PATHS = get_folderpaths_in_dir(MODEL_FOLDER_PATH)
        TRAIN_RUN_NAMES = [path.name for path in TRAIN_RUN_PATHS]

        return (TRAIN_RUN_NAMES, TRAIN_RUN_PATHS)
    

    def setup_on_run(self, run_path):
        """this function set's up this object to analyze a specific run, based
        on a run's folder. This folder will contain the h5-model and a config
        file with all information needed to initialise data etc.

        Args:
            run_path (PosixPath): path to the desired run's folder
        """
        # assign the runs name
        self.RUN_NAME = run_path.name
        # get the models configs
        self.PATH_TO_MODEL_CONFIG = self.get_model_config_path(run_path)
        # load the config dict from the run
        self.model_config = simple_model_config_df_2_dict_converter(
            df_from_model_config(
                self.PATH_TO_MODEL_CONFIG
            )
        )
        # get the sequence len and fps reduction fac of the model used
        self.SEQ_LEN = int(
            self.model_config["sequence_len"]
        )
        self.FPS_REDUCTION_FAC = int(
            self.model_config["fps_reduce_fac"]
        )
        # get fps of the input video from the config file
        # NOTE: the framerate of the videodata used for training needs to be
        #       the same as for the evaluated here. Otherwise, the dynamic
        #       calculation of the output framerate (in the motion_tracking_
        #       helpers.py Script) will lead to slower or fastened output videos
        self.ORIGINAL_FPS = float(
            self.model_config["input_framerate"]
        )

        # load the trained model
        self.model = self.get_model(run_path)

    def get_model_config_path(self, run_path):
        """return path to the config file of the model used

        Args:
            run_path (PosixPath): path to the desired run's folder
        
        Returns:
            PATH_TO_MODEL_CONFIG (PosixPath): the path to the file
        """
        CONFIG_FILE_NAME = name_constants("CONFIG_FILE_NAME")
        PATH_TO_MODEL_CONFIG = Path(
            f"{run_path}/" \
            f"{CONFIG_FILE_NAME}"
        )
        return (PATH_TO_MODEL_CONFIG)

    def get_model(self, run_path):
        """load the trained model which shall be used for evaluation/shall
        be evaluated

        Args:
            run_path (PosixPath): path to the desired run's folder

        Returns:
            model (tf.model): the trained model
        """
        run_cont = get_filepaths_in_dir(run_path)
        # NOTE: this could also get saftey checked, e.g. if there is only one
        # 		h5 model in etc.
        model_path = [path for path in run_cont if ".h5" in str(path)][0]
        # load the model
        # NOTE: the custom_layers.py script needs to be imported while this
        #       script runs, otherwise the loader cannot decode the custom
        #       layers used
        model = keras.models.load_model(model_path)
        return (model)
    
    def add_information(self, model_name, data_name, **kwargs):
        """add informations about a model's performance on a datapack to the
        objects central memory. These informations can be added during loop-
        computations that are run anyway and can later be used to compare
        several combinations, without having to recompute everything

        Args:
            model_name (str): the name of a run
            data_name (str): the dataset which was analysed
            **kwargs: pass the metrics that shall be saved, like the accuracy,
                      f1-scores or a confusion matrix
        """
        # create or assign the model-data combinations storage
        try:
            # test if it already exists
            exists = self.eval_storrage[model_name][data_name]["exists"]
        except:
            try:
                self.eval_storrage[model_name][data_name] = {}
            except:
                self.eval_storrage[model_name] = {}
                self.eval_storrage[model_name][data_name] = {}
        save_dict = self.eval_storrage[model_name][data_name]
        save_dict["exists"] = True
        
        # check if any keys already exist and make sure in this case feedback is
        # provided, as one would expect that this should not happen
        already_existent_keys = (
            list(filter(lambda x: x in save_dict.keys(), kwargs.keys()))
        )
        if len(already_existent_keys) > 0:
            print(f"WARNING\nrun     : '{model_name}'\ndatapack: '{data_name}'")
            print("Overwriting value for key(s): ")
            for key in already_existent_keys:
                print(f"'{key}' ", end="")
            print()

        # save the given args
        save_dict.update(kwargs)


def simple_model_config_df_2_dict_converter(config_df):
    """convert the dataframe from a config file to the dictionary it was, before
    it was written to the csv file. 
    The function recovers the shape and gets back the original datatypes, as all
    variables are strings when they are read from the csv file

    Args:
        config_df (pd.DataFrame): the dataframe, derived from the config-csv
                                  file of a run. this config_df is simply the
                                  result of the function from this script:
                                  df_from_model_config(configfile_path)
    
    Returns:
        config_dict (dict): the config dict that is just like the one originally
                            used by the architecture scripts
    """
    # convert to original dict structure
    pre_config_dict = config_df.T.to_dict()[1]
    # convert back to the right datatypes, which are in 
    # [Bool, list(str), int, float, str]
    config_dict = {}
    for key, val in pre_config_dict.items():
        # error variable to ensure the value gets detected by the structure
        # below
        value_detected = False
        # this section should convert everything safely. Espacially the problem,
        # that if a local variable has the same name as a string that gets
        # analysed here, is avoided by the approach below. If one would simply
        # wrap the eval() function in a try&except block, local variable values
        # might be inserted for some strings
        try:
            # check if the string can be converted to a float, this will catch
            # all numeric values
            _ = float(val)
            value = eval(val)
            value_detected = True
        except:
            # catch Booleans
            if val in ["True", "False"]:
                value = eval(val)
                value_detected = True
            else:
                try:
                    # this will get lists
                    if isinstance(eval(val), list):
                        value = eval(val)
                        value_detected = True
                # now only real string should be left, so assign them
                except:
                    if isinstance(val, str):
                        value = val
                        value_detected = True
        if value_detected == True:
            # assign the converted value to the config_dict
            config_dict[key] = value
        else:
            # show an error if none of the above conversions got triggered
            raise ValueError(
                f"Unable to convert value from key: '{key}' properly. " \
                f"Value is: '{val}'. If a new datatype was introduced for " \
                f"the config dicts, rework the functino that throws this " \
                f"error."
            )
    return(config_dict)


def pickle_anything(obj, name, path):
    """pickle any desired object
    
    Arguments:
        obj (whatever): the thing to be pickled
        name (str): the name with which the thing gets saved
        path (PosixPath): the path where the thing gets saved
    """
    with open(path / f"{name}.pickle", 'wb') as handle:
        # use protocol 4 instel of HIGHEST_PROTOCOL (==5), to enlage
        # compatibility
        pickle.dump(obj, handle, protocol=4)


def load_pickeled(name, path):
    """load privious stored objects
    
    Arguments:
        name (str): the name with which the thing gets saved
        path (PosixPath): the path where the thing got saved
    
    Returns:
        obj (whatever): the loaded object
    """
    with open(path / f"{name}.pickle", 'rb') as handle:
        obj = pickle.load(handle)
    return(obj)


if __name__ == "__main__":
    pass
