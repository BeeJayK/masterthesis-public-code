#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:06:20 2021

@author: marlinberger

Routine to create a ready-to-train environment.
Based on the folders: "00_Raw_Inputs", "01_Renamed_Inputs" & "02_Label_Tables",
it will create the structure of "03_Train_Data".

Description of the content in the folders:
	"00_Raw_Inputs"
		Contains the output from the video-camera. At the moment, there shall
		be a folder for every video, containing the video as collection of tiff
		pictures
	
	"01_Renamed_Inputs":
		The structure here will be a mirror of "00_Raw_Video", but with the
		inputs renamed to ensure they are named properly: The Name of every file
		within each video folder now represents its frame number, starting at 0.
		The contents in there is to be used for the labeling

	"02_Label_Tables"
		This table is to be created for every video, this is the labeling
		process. For every folder in "00_Raw_Inputs", one file is to be found
		in this directory.
		Every file contains a plain textfile, where each line contains the
		startframe of a motion, together with its motion-class, delimited with
		a comma.
		e.g.:
			0,0
			21,1
			45,0
			70,3
		This file would mean:	From frame 0 to frame 20, motion class is 0
								From frame 21 to frame 44, motion class is 1
								From frame 45 to frame 69, motion class is 0
								From frame 70 to end 20, motion class is 3
		Every .tab-file needs to start with 0

		"03_Train_Data"
			contains one folder for every video in "00_Raw_Inputs". Inside every
			video folder, a binary-file can be found for every frame of the
			video. This file has the same structure as in the UMH-mqtt-stack
			scenario. It contains the unprocessed mediapipe-hands output, allong
			with the right-left-hand evaluation from mediapipe-handness
			extension.
			The filename has the form [framenum]_[motion_class] and therefore
			can be used effectively in each models input pipeline
"""
# python packages
import cv2
import os
from pathlib import Path
import shutil
import sys
from tqdm import tqdm

# own packages
from motion_tracking_helpers import motion_tracking_helpers as mt_hlp


# USER INPUTS
# ----------------
def initialise():
    """user inputs: in which mode this script shall run

    Returns:
        MODE (int): 0 == Rename the "00_Raw_Inputs"-files with numbers
                         representing the framenumber in the video from 0 to x
                    1 == Create the train-ready data
    """
    # determine the mode in which the script shall run
    global MODE
    MODE = 1  # in [0, 1]
    return (MODE)


# ----------------


class Folder_Specs():
    """an object of this class will contain information about the folder
    structure etc., which are needed at different places in the script
    """

    def __init__(self, MODE, raw_input="video"):
        """
        Args:
            MODE (int): 0 == Rename the "00_Raw_Inputs"-files with numbers
                             representing the framenumber in the video from
                             0 to x
                        1 == Create the train-ready data
            raw_input (str): in ["pics", "video"]; determine the format of the
                             raw inputs
        """
        # in which mode is the current script runnning
        self.MODE = MODE

        # save trigger settings
        self.raw_input = raw_input

        # scan directorys, get paths to input videos
        self.INPUT_VIDEO_PATHS = self.get_input_video_paths()

        # analyse the renamed-data-dir
        self.RENAMED_INPUT_VIDEOS_PATHS, self.RENAMED_INPUT_DIR_EMPTY = (
            self.get_renamed_input_infos())

        # analyse the labels-table-dir
        self.LABEL_TABLES_PATHS, self.LABEL_TABLES_DIR_EMPTY = (
            self.get_label_tables_infos())

        # check if the train-data-dir is empty
        self.TRAIN_DATA_DIR_EMPTY = self.get_train_data_dir_infos()

    def get_input_video_paths(self):
        """get the paths of all input-video-folders / input-videos

        Returns:
            input_video_paths (list(PosixPath)): Paths to all input-video-
                                                 folders or video (depending on
                                                 the raw input type)
        """
        # path to the raw_videos main folder
        input_videos_main_path = Path(
            mt_hlp.name_constants("RAW_VIDEO_DIR_NAME")
        )
        if self.raw_input == "pics":
            # scan for folder for subfolder that contain the pics per video
            input_video_paths = mt_hlp.get_folderpaths_in_dir(
                input_videos_main_path
            )
        elif self.raw_input == "video":
            # scan the toplevel folder and exclude all files, that are no videos
            pre_paths = mt_hlp.get_filepaths_in_dir(
                input_videos_main_path
            )
            input_video_paths = [
                x for x in pre_paths if ".mov" in x.name
            ]
        return (input_video_paths)

    def get_renamed_input_infos(self):
        """Get paths of the folders in the renamed-inputs-directory.
        Also check if there are any folder at all. This is used as safty check
        for the input-rename-process and ensures nothing gets lost if the script
        is used by anyone who doesnt know what it does

        Returns:
            folderpaths (list(PosixPath)): paths to the folders in the renamed
                                           inputs dir
            renamed_input_dir_is_empty (bool): as it's name say...
        """
        # get content in the renamed-inputs-dir
        renamed_input_dir_path = Path(
            mt_hlp.name_constants("RENAMED_INPUTS_DIR_NAME")
        )
        folderpaths = mt_hlp.get_folderpaths_in_dir(renamed_input_dir_path)
        # if nothing is inside, it is empty -> True
        if len(folderpaths) == 0:
            renamed_input_dir_is_empty = True
        else:
            renamed_input_dir_is_empty = False
        return (folderpaths, renamed_input_dir_is_empty)

    def get_label_tables_infos(self):
        """Get paths of the label-tables in the label-tables-directory

        Returns:
            tablepaths (list(PosixPath)): paths to all label-tables
            label_tables_dir_is_empty (bool): as it's name say...
        """
        # get content in the label-tables-dir
        label_tables_dir_path = Path(
            mt_hlp.name_constants("LABEL_TABBLE_DIR_NAME")
        )
        filepaths = mt_hlp.get_filepaths_in_dir(label_tables_dir_path)

        # fetch suffix infos for the label tables
        tab_suffix = mt_hlp.name_constants("TABLES_SUFFIX")
        # get paths to label-tables only
        tablepaths = [path for path in filepaths if
                      path.name[-len(tab_suffix):] == tab_suffix]

        # if nothing is inside, it is empty -> True
        if len(tablepaths) == 0:
            label_tables_dir_is_empty = True
        else:
            label_tables_dir_is_empty = False
        return (tablepaths, label_tables_dir_is_empty)

    def get_train_data_dir_infos(self):
        """Check if there are any folder in the train-data-directory. This is
        used as safty check before new folder will be created and ensures
        nothing gets lost if the script is used by anyone who doesnt know what
        it does.
        NOTE: routine could be changed to detect if the videonames already
              exist. But as this is an internal script, this seems a little
              over the top to me (marlinberger)

        Returns:
            train_data_dir_is_empty (bool): as it's name say...
        """
        # get content in the renamed-inputs-dir
        train_data_dir_path = Path(mt_hlp.name_constants("TRAIN_DATA_DIR_NAME"))
        folderpaths = mt_hlp.get_folderpaths_in_dir(train_data_dir_path)
        # if nothing is inside, it is empty -> True
        if len(folderpaths) == 0:
            train_data_dir_is_empty = True
        else:
            train_data_dir_is_empty = False
        return (train_data_dir_is_empty)


def rename_pics_to_start_at_zero(Folder_Specs):
    """ensure that in all raw-video-input folders, the pictures are
    named acending, starting at 0. Save them with theire new names in the
    RENAMED_INPUTS_DIR

    Args:
        Folder_Specs (obj): Contains folder structur information an methods
    """
    # itterate over all input videos
    raw_input_dir = mt_hlp.name_constants("RAW_VIDEO_DIR_NAME")
    print(f"Itterating over videos in '{raw_input_dir}'")
    for video_path in tqdm(Folder_Specs.INPUT_VIDEO_PATHS):
        # routine if the raw inputs are already extracted pictures
        if Folder_Specs.raw_input == "pics":
            # get paths to all frames. hidden system files are excluded
            pics = mt_hlp.get_filepaths_in_dir(video_path)
            # naturally sort them
            pics_sorted = sorted(pics)

            # itterate over pictures, rename them ascending, save them under
            # the new destination
            for framenum, pic_path in enumerate(tqdm(pics_sorted, leave=False)):
                # extract file name, suffix and parent directory from the input
                # file
                original_name = pic_path.stem
                original_extension = pic_path.suffix
                video_name = pic_path.parent.name

                # create the new filename, by attaching the framenumber to the
                # front of the original filename
                new_name = f"{framenum}_{original_name}{original_extension}"

                # create the path to the destination, where the file with the
                # new name shall be placed at
                renamed_file_path = Path(
                    f"{mt_hlp.name_constants('RENAMED_INPUTS_DIR_NAME')}/" \
                    f"{video_name}/" \
                    f"{new_name}"
                )

                # create the new directory if it does not already exist
                os.makedirs(os.path.dirname(renamed_file_path), exist_ok=True)
                # copy the input file with its new name, that contains the
                # ascending framenumber in the beginning
                shutil.copy(pic_path, renamed_file_path)

        # if the raw inputs are videos, the tiff's need to be extracted
        elif Folder_Specs.raw_input == "video":
            # initialise video cap, frame number count and videoname
            vidcap = cv2.VideoCapture(str(video_path))
            fn_count = 0
            video_name = "".join(str(video_path.name).split(".")[0])

            # initialise the tqdm bar manually
            length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm(total=length + 1, leave=False)

            # read first image
            success, image = vidcap.read()
            while success:
                # save frame as tiff file
                new_name = f"{fn_count}_IMG_{video_name}.tif"
                renamed_file_path = Path(
                    f"{mt_hlp.name_constants('RENAMED_INPUTS_DIR_NAME')}/" \
                    f"{video_name}/" \
                    f"{new_name}"
                )
                os.makedirs(os.path.dirname(renamed_file_path), exist_ok=True)
                cv2.imwrite(str(renamed_file_path), image)

                # read next pic here, to exit if the read fails (which is caused
                # if the end of the video is reached)
                success, image = vidcap.read()

                # update frame count and tqdm bar
                fn_count += 1
                pbar.update(1)


def safty_checks_before_train_data_creation(Folder_Specs):
    """check if everything fits, if not, tell the user whats wrong. Extend this
    function with more checks if desired

    Args:
        Folder_Specs (obj): Contains folder structur information an methods

    Returns:
        all_good (bool): indicate if everythings all right
        error_str (str): used to collect all failed safty check and print them
                         to the person using the script. this way, all errors
                         can be signaled at once
    """
    # initialise the return param
    all_good = True
    error_str = ""

    # check if there exists a labels-table for every video in the
    # renamed-inputs-dir
    videonames = [name.name for name in Folder_Specs.RENAMED_INPUT_VIDEOS_PATHS]
    label_table_names = [name.name for name in Folder_Specs.LABEL_TABLES_PATHS]
    # concetenate all label-table-names, check agains this combination in the
    # loop, if every video can be assigned to a label table
    combined_label_table_names = "".join(label_table_names)
    for videoname in videonames:
        if videoname not in combined_label_table_names:
            error_str += (
                f"\nlabel table not found for video: '{videoname}'\n"
            )

    # check if the train-data-dir is empty
    if not Folder_Specs.TRAIN_DATA_DIR_EMPTY:
        error_str += (
            f"\ndirectory: '{mt_hlp.name_constants('TRAIN_DATA_DIR_NAME')}' " \
            f"already contains folders, which is not allowed\n"
        )

    # check if the movement class assignment in the label-tables for each
    # video in the renamed-inputs-dir, starts at 0
    for label_table_path in Folder_Specs.LABEL_TABLES_PATHS:
        # open the label table
        table_lines = mt_hlp.open_labels_table(label_table_path)
        # check if the first entry for the framenumber is a zero
        if table_lines[0][0] != "0":
            error_str += (
                f"\nlabel-table: '{label_table_path.name}' " \
                f"does not start at frame '0', which is mandatory\n"
            )

    # check that every line is splittable by an "," and the outcomes are
    # convertable to integers. If not, return the error already here, as
    # different next tests will otherwise through errors
    for label_table_path in Folder_Specs.LABEL_TABLES_PATHS:
        # open the label table
        table_lines = mt_hlp.open_labels_table(label_table_path)
        # itterate through the lines
        for line in table_lines:
            try:
                # try splitting and converting them to integers
                _ = int(line.split(",")[0])
                _ = int(line.split(",")[1])
            except:
                error_str += (
                    f"\nlabel-table: '{label_table_path.name}', line: " \
                    f"'{line}' can not be splited by a ','; and be" \
                    f"converted to an integer, which is mandatory\n"
                )
                all_good = False
                error_str = mt_hlp.adjust_error_message(error_str)
                return (all_good, error_str)

    # check that all frame numbers, given in the label tables, are ascending
    for label_table_path in Folder_Specs.LABEL_TABLES_PATHS:
        # open the label table
        table_lines = mt_hlp.open_labels_table(label_table_path)
        # extract the framenumbers as list of ints
        framenums = [int(num.split(",")[0]) for num in table_lines]
        # check that they are always bigger than the framenumber before
        strict_ascending = True
        for framenum_0, framenum_1 in zip(framenums[:-1], framenums[1:]):
            if framenum_0 >= framenum_1:
                # safe if they fail the test
                strict_ascending = False
        # write error_str if necessary
        if not strict_ascending:
            error_str += (
                f"\nframenumbers in label-table: '{label_table_path.name}' " \
                f"do not strictly increase, which is mandatory\n"
            )

    # check that the highest appearing framenumber in each label table is not
    # higher than the number of pictures available
    for label_table_path in Folder_Specs.LABEL_TABLES_PATHS:
        # open the label table
        table_lines = mt_hlp.open_labels_table(label_table_path)
        # get the highest framenumber
        highest_framenum_labeled = int(table_lines[-1].split(",")[0])
        # get the corresponding videoname
        tab_suffix_len = len(mt_hlp.name_constants("TABLES_SUFFIX"))
        videoname = label_table_path.name[:-tab_suffix_len]
        # get the number of images for the video
        got_path = False
        for path in Folder_Specs.RENAMED_INPUT_VIDEOS_PATHS:
            # assign the needed path
            if path.name == videoname:
                path_to_inspect = path
                # signal that the path was found
                got_path = True
                break
        # only analyse the the path, if an assignment was found. if a video
        # has no corresponding labels-table, this case is already catched above.
        # the case that there is no videodata but a labels table is really
        # obvious and wont be checked by the code, also as if this case doesnt
        # cause trouble...
        if got_path:
            content = mt_hlp.get_filepaths_in_dir(path_to_inspect)
            if highest_framenum_labeled > len(content):
                # shorten the calls
                re_inp_dir = mt_hlp.name_constants("RENAMED_INPUTS_DIR_NAME")
                # write the error message
                error_str += (
                    f"\nFor label-table: '{label_table_path.name}', " \
                    f"the highest framenumber labeled is greater than the " \
                    f"number of images in '{re_inp_dir}/{videoname}', which " \
                    f"is not allowed\n"
                )

    # check that only valid characters are used in the label tables
    # determine which content is allowed in the tables (only numbers and ',')
    allowed_content = [str(i) for i in range(10)]
    allowed_content.append(",")
    # itterate through all label tables
    for label_table_path in Folder_Specs.LABEL_TABLES_PATHS:
        # open the label table
        table_lines = mt_hlp.open_labels_table(label_table_path)
        # merge them to one string, which makes it easy and fast to analyse
        merged_table_content = "".join(table_lines)
        # itterate throuh all entrys in the table
        for check_character in merged_table_content:
            # if a forbidden character is found, write the error message
            if check_character not in allowed_content:
                error_str += (
                    f"\nlabel-table: '{label_table_path.name}' " \
                    f"contains the following forbidden character: " \
                    f"'{check_character}'\n"
                )

    # if the error_str contains anything, set all_good-flag to false to not
    # start the train data creation. add intro line to error_str
    if len(error_str) != 0:
        all_good = False
        error_str = mt_hlp.adjust_error_message(error_str)

    return (all_good, error_str)


def create_train_ready_data(Folder_Specs):
    """this function is to be triggered, if all safty changed were passed. It
    will create the ready-to-train, labeled data, based on the content in the
    input- and labelfolders. See descriptions of them above.

    Args:
        Folder_Specs (obj): Contains folder structur information an methods
    """
    # get the paramset, with which all pictures shall be evaluated by mediapipe
    paramset = mt_hlp.mediapipe_hands_paramsets(0)

    # itterate over all videos
    renamed_input_dir = mt_hlp.name_constants("RENAMED_INPUTS_DIR_NAME")
    print(f"Itterate over videos in: '{renamed_input_dir}'")
    for video_path in tqdm(Folder_Specs.RENAMED_INPUT_VIDEOS_PATHS):
        # get paths to all frames. hidden system files are excluded
        pics = mt_hlp.get_filepaths_in_dir(video_path)
        # sort the renamed pictures by frame number
        pics_sorted = mt_hlp.sort_paths_by_first_num(pics)

        # get video_name to safe files in this folder in the train data dir
        video_name = video_path.name

        # create the folder for the video in the train-data-dir
        os.makedirs(os.path.dirname(Path(
            f"{mt_hlp.name_constants('TRAIN_DATA_DIR_NAME')}/{video_name}/_")),
            exist_ok=True)

        # load the labels table. As the paths are now in the Folder_Specs-obj,
        # this could also be used for access
        label_tab_path = Path(
            f"{mt_hlp.name_constants('LABEL_TABBLE_DIR_NAME')}/" \
            f"{video_name}{mt_hlp.name_constants('TABLES_SUFFIX')}"
        )

        # get the motion class vector
        motionclass_vector = mt_hlp.get_motionclass_vector(
            label_tab_path, len(pics_sorted)
        )

        # itterate over the ascending renamed pictures pictures
        for framenum, (pic_path, motionclass) in enumerate(tqdm(
                zip(pics_sorted, motionclass_vector),
                total=len(pics_sorted),
                leave=False)):
            # get the numpy representation of the picture
            image = mt_hlp.load_tiff_image(pic_path)

            # process the image
            results = mt_hlp.mediapipe_hands_process(image, paramset)

            # combine framenumber and motionclass to name the file
            # TODO: CameraID mit speichern
            save_name = f"{framenum}_{motionclass}"

            # create the filename and save-path
            file_path = Path(
                f"{mt_hlp.name_constants('TRAIN_DATA_DIR_NAME')}/" \
                f"{video_name}/{save_name}"
            )

            # process the mp results to the shape that is used all across the
            # project
            structured_output = mt_hlp.convert_mediapipe_output_v2(results)

            # write the file
            mt_hlp.write_coordinates_to_csv(file_path, structured_output)


if __name__ == "__main__":

    # get the mode in which the script shall run
    MODE = initialise()
    # get the relevant folder structure stuff
    Folder_Specs = Folder_Specs(MODE)

    # Rename-the-raw-input mode
    if MODE == 0:
        if Folder_Specs.RENAMED_INPUT_DIR_EMPTY:
            # if everything alright, start the renaming
            rename_pics_to_start_at_zero(Folder_Specs)
        else:
            # if so, print whats wrong
            error_in = mt_hlp.name_constants('RENAMED_INPUTS_DIR_NAME')
            print(f"\nWarning\n\tDoesnt start renaming.\n\tDirectory: " \
                  f"'{error_in}'\n\talready contains folders\n")

    # Create-the train-ready-data mode
    if MODE == 1:
        # perform safty checks
        all_good, error_str = safty_checks_before_train_data_creation(
            Folder_Specs)

        # create the train-ready data, if all saftey checks were passed
        if all_good:
            create_train_ready_data(Folder_Specs)
        # print error messages if any appered
        else:
            print(error_str)
