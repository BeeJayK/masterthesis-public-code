#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:06:20 2021

@author: marlinberger

Testscript to test tf functions and basic concepts to implement in custom
layers and stuff
"""
# python packages
from re import S
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

# own pakages
from motion_tracking_helpers import coordinate_position_assignment as cpa
from motion_tracking_helpers import custom_layers as cus_lay

# own modules
from _def_snippits import ExampleTensors as ExT



#run = api.run("beejayk/MotionClassification_DCC/driven-sweep-76")

import pandas as pd
import wandb

api = wandb.Api()
entity, project = "beejayk", "MotionClassification_DCC"
runs = api.runs(entity + "/" + project)
sweeps = api.sweeps(entity + "/" + project)

for sweep in sweeps:
    print(sweep.name)


summary_list, config_list, name_list, id_list = [], [], [], []
for i, run in enumerate(runs):
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

    # save the runs id for further api communication
    id_list.append(run.id)

    if i == 5:
        break

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list,
    "id": id_list
    })

print(runs_df)