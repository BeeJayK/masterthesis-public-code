#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:35:40 2021

@author: marlinberger

TODO: description text schreiben
"""
# python packages
import matplotlib.pyplot as plt
import numpy as np

# own packages
from motion_tracking_helpers import motion_tracking_helpers as mt_hlp
from motion_tracking_helpers import mt_neural_helpers as neu_hlp
from motion_tracking_helpers import mt_plot_helpers as plt_hlp
# this import is necessary sothat the custom_layers of the loaded model can be
# decoded
from motion_tracking_helpers import custom_layers

# User Inputs
# ------------
analyse_config = {
    "ANALYSE_FOLDER": "Thesis_example",#Multi_test, New_model_test, Dev_Promissing_Models

    #"DATA_PACKS_TO_ANALYSE": ["straight_80_20_val", "straight_80_20_train", "Holdout_Heiko", "Federico_straight_80_20_val", "Nils_straight_80_20_val"],#Nils_straight_80_20_val
    "DATA_PACKS_TO_ANALYSE": [
        "0_full_train", "0_full_val", "2_Dk_val", "2_Dq_val", "2_F_val",
        "2_J_val", "2_K_val", "2_M_val", "2_Nk_val", "2_Ns_val", "3_H_holdout"
    ],
    "DATA_PACKS_TO_ANALYSE": ["0_full_val"],

    "overview_name": "Thesis_0", # name is added to overview plots

    "MODE": 0, # in 0 (==save), 1 (=show), 2 (==save&show)

    "DEV_MODE": 0, # in 0 (==off), 1 (=work), 2 (==create pickled)
    "DEV_SAVE_NAME": "Thesis_0",#f1_dev_small, all_classes, full_dev_big

    "USE_ID_NAMES": False, # this will only affect the 'once'-analysations

    "ANALYSATIONS": {
        "per_model": {
            # print overview stuff in the terminal
            "TERMINAL_PRINTS": False,

            # confusion matrix
            "CONF_MATR_PLOT": True,
            "CONF_MATR_PLOT_NORM": True,
            "CONF_MATR_PLOT_wc": True,
            "CONF_MATR_PLOT_NORM_wc": True,

            # stuff like accuracy, recall, precision, f1-score
            "OVERVIEW_METRICS_AVG": True,

            # display per-class-insights (f-score, precision, recall, support)
            "F1_SCORE_SUPPORT_PER_CLASS": True,
            "PREC_RECALL_PER_CLASS": True,

            # accuracys over sample time
            "ACC_OVER_TIME": True,

            # inspect accuracy vs motion-class-trainsition
            "ACC_OVER_CLASS_APPEARANCE": True,

            # anaylse how many frames have detected hande/2_hands... das
            # irgendwie abbilden
            # NOTE: den auf jeden fall normiert machen: hands/samples oder so
            # TODO: not yet implemented
            "HANDS_CLASSIFIED" : False
        },
        "once": {
            # plot a matrix, showing model's and their accuracy's on datasets
            "MODELS_VS_DATA_ACC": False,

            # simularity between models, based on theire confusion matrices
            # NOTE possible is either a new matrix that showes scores,
            #      calculated based on the confusion matrixes, between models,
            #      OR with one resulting matrix for 2 models that shows
            #      conf_1-conf_2
            # TODO: not yet implemented
            "COMPARE_SIMULARITY": False,
            "COMPARE_MODELS": [
                ("run_1", "run_2"),
                ("run_1", "run_3")
            ]
        },
        "terminal_print": False
    }
}
# ------------

def plot_saver(f, plot_name, pass_args, base_path, overwiew=False,
        create_pickled=True):
    """function saves the plots in the desired places if saving is activated

    Args:
        f (matplotlib.figure): the figure to be saved
        plot_name (str): how the file of the saved plot shall be named
        pass_args (tuple): contains 3 arguments that describe how the given
                           predictions were derived:
                           run_name (str): the name of the run, in which
                                           the model has it's origin
                           datapack_name (str): the name of the datapack
                                                that's currently analysed
                           config (dict): config dict from the run
        base_path (PosixPath): the path to the folder in which every plot
                               gets saved
        overwiew (bool): wether the figure so save is from a per-model-and-
                         dataset analysation or from an overview. The save
                         strategys and namings are a bit different
        create_pickled (bool): wether the figure shall also be pickled or not.
                               pickeling might conflict with some custom
                               classes, therefore this option was added to save
                               the png's anyway
    """
    # set up everything for saving a per-model-and-dataset plot
    if not overwiew:
        # unpack the zipped values
        run_name, datapack_name, _ = pass_args
        # create the name with which the files are saved
        save_name = f"{plot_name}__{datapack_name}__{run_name}"
        # create the save path
        save_path = base_path / save_name
    else:
        # unpack the zipped values
        overview_name, _ , _ = pass_args
        # create the name with which the files are saved
        try_name = f"{plot_name}__{overview_name}"
        # stack together the path to try for saving
        try_path = base_path / (try_name)
        # get a unique version of the path
        save_path = mt_hlp.get_unique_path_variation(try_path)
        # assign the unique name as save name
        save_name = save_path.stem
    # save the figure as png, use same dpi as while displaying the figure
    f.savefig(save_path, dpi=f.dpi)
    # save the figure pickeled, to enable quick later adjustments
    if create_pickled:
        mt_hlp.pickle_anything(f, save_name, base_path)


def get_analysation_data(analyse_config, dataset_2_analyse, Eval_Specs):
    """depending on this and that, load the desired data for the current
    analysation run

    Args:
        analyse_config (dict): contains information on the desired plots and
                               analysations
        dataset_2_analyse (tf.dataset): the dataset to analyse
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder and the current run

    Returns:
        X (np.array): the underlaying x data
        Y_true (np.array): the labels for the whole dataset
        Y_pred (np.array): the predictions for the whole dataset
    """    
    # extract the labels and the predictions from the given dataset with the
    # given model
    if analyse_config["DEV_MODE"] in [0, 2]:
        X, Y_true, Y_pred = neu_hlp.get_predictions_from_ds(
            Eval_Specs.model,
            dataset_2_analyse,
            verbose=False
        )
        # possibly save the resulting data for developement purposes
        if analyse_config["DEV_MODE"] == 2:
            save_path = (
                mt_hlp.get_abs_script_path(__file__) /
                mt_hlp.name_constants("ANALYZE_DIR_NAME") /
                mt_hlp.name_constants("PICKLED_DEV_BLOCKS")
            )
            mt_hlp.pickle_anything(
                (X, Y_true, Y_pred), analyse_config["DEV_SAVE_NAME"], save_path
            )
    # load pickeled x and y's in developement mode, to enable faster developing
    elif analyse_config["DEV_MODE"] == 1:
        save_path = (
            mt_hlp.get_abs_script_path(__file__) /
            mt_hlp.name_constants("ANALYZE_DIR_NAME") /
            mt_hlp.name_constants("PICKLED_DEV_BLOCKS")
        )
        X, Y_true, Y_pred = mt_hlp.load_pickeled(
                analyse_config["DEV_SAVE_NAME"], save_path
            )
    
    return(X, Y_true, Y_pred)


def single_analyses(analyse_config, Eval_Specs):
    """perform all the analysations that are to be performed once per script-
    trigger and not itterative for every model and dataset. The functions here
    rely on values that got evaluated for every model and dataset in the
    previous itteration process

    Args:
        analyse_config (dict): contains information on the desired plots and
                               analysations
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder and the current run
    """
    # assign the save settings
    save_figs = Eval_Specs.save_figures
    base_path = Eval_Specs.save_base_path
    pass_args = (analyse_config["overview_name"], None, None)

   # possibly save or overwrite some Eval_Specs vars for developement purposes
    dev_save_path = (
            mt_hlp.get_abs_script_path(__file__) /
            mt_hlp.name_constants("ANALYZE_DIR_NAME") /
            mt_hlp.name_constants("PICKLED_DEV_BLOCKS")
        )
    if analyse_config["DEV_MODE"] == 2:
        mt_hlp.pickle_anything(
            (
                Eval_Specs.eval_storrage,
                Eval_Specs.fetched_run_data_available,
                Eval_Specs.fetched_run_data
            ),
            f"{analyse_config['DEV_SAVE_NAME']}_eval_pack",
            dev_save_path
        )
    elif analyse_config["DEV_MODE"] == 1:
        eval_pack = mt_hlp.load_pickeled(
                f"{analyse_config['DEV_SAVE_NAME']}_eval_pack", dev_save_path
            )
        Eval_Specs.eval_storrage = eval_pack[0]
        Eval_Specs.fetched_run_data_available = eval_pack[1]
        Eval_Specs.fetched_run_data = eval_pack[2]

    # plot the model-dataset-accuracy matrix
    if analyse_config["ANALYSATIONS"]["once"]["MODELS_VS_DATA_ACC"]:
        f = plt_hlp.models_vs_datasets_accuracy(Eval_Specs)
        if save_figs:
            plot_saver(
                f, "models_vs_datapacks_accuracy", pass_args, base_path,
                overwiew=True)


def analyse_distributer(analyse_config, Eval_Specs, dataset_2_analyse,
        datapack_name):
    """with a given model and dataset, perform all desired analysations

    Args:
        analyse_config (dict): contains information on the desired plots and
                               analysations
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder and the current run
        dataset_2_analyse (tf.dataset): the dataset to analyse
        datapack_name (str): the name of the datapack that's currently analysed
    
    Returns:
        Eval_Specs (obj): the input object, enhanced with some metrics for the
                          present model and data
    """
    # TODO: wrap das in ein try&except block. bekannte fehler sollten unten im
    #       error_check gecatcht werden, allerdings bezweifel ich dass dafür
    #       die zeit reicht. Also dafür sorgen dass ein OUTPUT kommt der mir
    #       beim debuggen hilft

    # TODO: die figures wieder schließen nach den funktionen falls sie nicht
    #       gezeigt werden sollen, da das wohl sonst viel RAM zieht
    
    # assign the save settings
    save_figs = Eval_Specs.save_figures
    base_path = Eval_Specs.save_base_path

    # get the data to analyse
    X, Y_true, Y_pred = get_analysation_data(
        analyse_config, dataset_2_analyse, Eval_Specs
    )

    # reduce the labels to the one with the max probability
    Y_true_argmax = np.argmax(Y_true, axis=1)
    Y_pred_argmax = np.argmax(Y_pred, axis=1)

    # combine some arguments that are mostly passed together
    Y_raw = (Y_true, Y_pred)
    Y_max_prob = (Y_true_argmax, Y_pred_argmax)
    pass_args = (Eval_Specs.RUN_NAME, datapack_name, Eval_Specs.model_config)

    # save essential information and metrics from the current model-data
    # combination to a central object be able to later per
    neu_hlp.save_analysation_data(Y_max_prob, pass_args, Eval_Specs)

    # print stuff to the terminal
    if analyse_config["ANALYSATIONS"]["per_model"]["TERMINAL_PRINTS"]:
        plt_hlp.terminal_prints(Y_max_prob)

    # plot the confusion matrix for the dataset
    if analyse_config["ANALYSATIONS"]["per_model"]["CONF_MATR_PLOT"]:
        f = plt_hlp.plot_conf_matr(
            Y_max_prob,
            pass_args
        )
        if save_figs:
            plot_saver(f, "conf_matr", pass_args, base_path)
    
     # plot the confusion matrix for the dataset with counter
    if analyse_config["ANALYSATIONS"]["per_model"]["CONF_MATR_PLOT_wc"]:
        f = plt_hlp.plot_conf_matr(
            Y_max_prob,
            pass_args,
            count=True
        )
        if save_figs:
            plot_saver(f, "conf_matr_w_count", pass_args, base_path)
    
    # plot the normalized confusion matrix for the dataset
    if analyse_config["ANALYSATIONS"]["per_model"]["CONF_MATR_PLOT_NORM"]:
        f = plt_hlp.plot_conf_matr(
            Y_max_prob,
            pass_args,
            normalize=True
        )
        if save_figs:
            plot_saver(f, "conf_matr_norm", pass_args, base_path)
    
     # plot the normalized confusion matrix for the dataset with counter
    if analyse_config["ANALYSATIONS"]["per_model"]["CONF_MATR_PLOT_NORM_wc"]:
        f = plt_hlp.plot_conf_matr(
            Y_max_prob,
            pass_args,
            normalize=True,
            count=True
        )
        if save_figs:
            plot_saver(f, "conf_matr_norm_w_count", pass_args, base_path)

    # plot some general model metrics
    if analyse_config["ANALYSATIONS"]["per_model"]["OVERVIEW_METRICS_AVG"]:
        f = plt_hlp.plot_metrics(
            Y_max_prob,
            pass_args,
        )
        if save_figs:
            plot_saver(f, "overview_metrics", pass_args, base_path)

    # plot overview-per-class metrics (f1-score and support)
    if analyse_config["ANALYSATIONS"]["per_model"][
            "F1_SCORE_SUPPORT_PER_CLASS"]:
        f = plt_hlp.plot_f1_score_support_per_class(
            Y_max_prob,
            pass_args,
        )
        if save_figs:
            plot_saver(
                f, "f1_score_and_support_per_class", pass_args, base_path
            )
    
    # plot more in-depth-per-class metrics (precision and recall)
    if analyse_config["ANALYSATIONS"]["per_model"]["PREC_RECALL_PER_CLASS"]:
        f = plt_hlp.plot_precision_and_recal_per_class(
            Y_max_prob,
            pass_args,
        )
        if save_figs:
            plot_saver(
                f, "precision_and_recall_per_class", pass_args, base_path
            )
    
    # plot in depth information about class accuracy's differing over the sample
    if analyse_config["ANALYSATIONS"]["per_model"]["ACC_OVER_TIME"]:
        f = plt_hlp.plot_acc_over_sample_time(
            Y_max_prob,
            pass_args,
        )
        if save_figs:
            plot_saver(
                f, "accuracys_over_sample_time", pass_args, base_path
            )
    
    # plot information about class accuracys relative to the action appearance
    if analyse_config["ANALYSATIONS"]["per_model"]["ACC_OVER_CLASS_APPEARANCE"]:
        f = plt_hlp.plot_acc_over_class_appearance(
            Y_max_prob,
            pass_args,
        )
        if save_figs:
            plot_saver(
                f, "accuracys_over_class_appearance", pass_args, base_path,
                create_pickled=False
            )
    
    return(Eval_Specs)


def analyze_outer_loop(analyse_config, Eval_Specs):
    """helper function, that loops through all conbinations of models and
    datasets that are to be evaluatet, initialises the Eval_Specs-object for
    the current configuration and passes these gathered elemtns to the function
    that will perform the analysations and plots for the given combination

    Args:
        analyse_config (dict): contains information on the desired plots and
                               analysations
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder and the current run
    """
    # set up the path that is needed to grant the DS_Builder access to the
    # data packs
    # TODO: das irgendwie besser lösen?!!! Direkt im Databuilder! Wobei hat
    #       der den path maschinenunabhängig?! weil das ist ja ein anderes
    #       package wo das liegt, also glaube geht eig nicht. höchstens mit
    #       init-magic beim import, aber das finde ich auch sketchy und das wäre
    #       absolut nur für meine anwendungen safe & hart zu debuggen wenn wer
    #       anderes damit arbeitet
    abs_path = (
        mt_hlp.get_abs_script_path(__file__) /
        mt_hlp.name_constants('ARCHITECTURE_DIR_NAME')
    )

    # fetch wandb-API information if the id_names should be used in the
    # comparison plots. The id's are significantly shorter - but also more
    # cryptic - than the run names
    if analyse_config["USE_ID_NAMES"]:
        # as read timeouts might occure, wrap this up to ensure plotting is
        # performed anyway but with the 'readable' run names
        try:
            wandb_runs_df = neu_hlp.fetch_wandb_project_data(
                "MotionClassification_DCC"
            )
            # include it in the Eval_Specs
            Eval_Specs.fetched_run_data = wandb_runs_df
            Eval_Specs.fetched_run_data_available = True
        except Exception as e:
            # print error, continue plotting but not use the id's while plotting
            print("Error while fetching the wandb data\n")
            print(f"message: {e}")
    
    # check if figures shall be saved and possibly create the saving path
    if analyse_config["MODE"] in [0, 2]:
        # activate the save-fig-flag
        save_figs = True
        # get the path of this script
        base_path = (
            mt_hlp.get_abs_script_path(__file__) /
            mt_hlp.name_constants("ANALYZE_DIR_NAME") /
            mt_hlp.name_constants("PLOT_SAVES")
        )
    else:
        save_figs = False
        base_path = "undefined"
    # update Eval_Specs
    Eval_Specs.save_figures = save_figs
    Eval_Specs.save_base_path = base_path

    # itterate over the models
    for run_path in Eval_Specs.TRAIN_RUN_PATHS:
        # initialise the Eval_Specs-object on the current run
        Eval_Specs.setup_on_run(run_path)
        # get the config dict from the run, which is needed for many thing, e.g.
        # preparing the datasets
        model_config = Eval_Specs.model_config
        # itterate over the datapacks
        for datapack_name in analyse_config["DATA_PACKS_TO_ANALYSE"]:
            # signal where the process is
            print(f"\n\nAnalyse\nModel: {Eval_Specs.RUN_NAME}\n" \
                  f"Data:  {datapack_name}")
            # only create dataset if no pickeled data shall be used (in dev
            # mode)
            if analyse_config["DEV_MODE"] in [0, 2]:
                # add the data_pack to be analyed to the model's config dict
                model_config["analyse_data_pack"] = datapack_name
                # build the dataset that's to be analyzed
                DS_Builder = neu_hlp.DataBuilder_v2(
                    model_config, abs_path, shuffle=False
                )
                dataset_2_analyse = DS_Builder.build("analyse")
            # if pickeled data is used, simply pass a dummy value for the ds
            if analyse_config["DEV_MODE"] == 1:
                dataset_2_analyse = False
            # go through the analysations that are to be done with every model
            Eval_Specs = analyse_distributer(
                analyse_config, Eval_Specs, dataset_2_analyse, datapack_name)
    
    # perform all analysations that are performed once per script-start
    single_analyses(analyse_config, Eval_Specs)


def error_checks(analyse_config):
    """function to ensure errors are detected and printed

    Args:
        analyse_config (dict): contains information on the desired plots and
                               analysations

    Returns:
        all_good (bool): indicate if everythings all right
        error_str (str): used to collect all failed safty check and print them
                         to the person using the script. this way, all errors
                         can be signaled at once
    """
    # initialise the return param
    all_good = True
    error_str = ""

    # if the error_str contains anything, set all_good-flag to false to not
    # start the train data creation. add intro line to error_str
    if len(error_str) != 0:
        all_good = False
        error_str = mt_hlp.adjust_error_message(error_str)

    return (all_good, error_str)


if __name__ == "__main__":

    # collect all information about the choosen evaluation folder set
    Eval_Specs = mt_hlp.Eval_Specs(
        analyse_config["ANALYSE_FOLDER"], MODE="Analyzer"
    )

    # perform error checks
    all_good, error_str = error_checks(Eval_Specs)

    # start the evaluation if everything is ok, or else print whats wrong
    if all_good:
        analyze_outer_loop(analyse_config, Eval_Specs)
    else:
        # print whats wrong if so
        print(error_str)
    
    # make plots stay open
    if analyse_config["MODE"] in [1, 2]:
        plt.show()
