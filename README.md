# Overview
This repository contains all the code developed during the handed masters thesis "Entwicklung eines Deep Learning Modells zur Analyse von Arbeitsschritten in einem manuellen Verpackprozess", which translates to "Devoloping a deep learning model to analyse work steps in a manual packaging process".

Scripts directly referenced in the thesis can be found in the folder '_0-code_showcases'.

These are:
- Create_Train_Data.py (creation of train-ready-data from raw video inputs)
- mt_neural_helpers.py (DataBuilder, Inhibiting functions, Callbacks and related functions)
- custom_layers.py (implementations of the (ML) preprocessing pipeline via custom layers)
- Conv_1D_Basic.py (expamle of the Pipeline for the Conv-1d Models)
- motion_tracking_helpers.py (general helper functions)


# Introduction
This repository is meant to give insights on the code, that I developed to solve the tasks of my master's thesis.

The thesis is done at the Digital Capibility Center (DCC) Aachen. The DCC is a cooperation between an institut of the RWTH Aachen and McKinsey, with the goal to bring digitalisation and AI into the production industry.

The work topic to be solved with the thesis is the classification of motion classes, performed by a worker, during the packaging process in a production line, based on a video stream.

The stream is processed by two neural networks, of which the first extracts the handcoordinates from images (using google's mediapipe) and the second takes these coordinate representations along with aditional information and predicts the current motion class.

A visualized models output look like:
<img width="1363" alt="github-repo-main-visualization" src="https://user-images.githubusercontent.com/75037677/162510168-8280dfcc-5d0c-43a4-b02c-46533ff4fb4f.png">


Based on reliable motion class predictions (action recognition), one can easily develop toplevel routines, that can solve tasks which are currently time consuming, complicated and expensive. These tasks include (dynamic) cycle time calculations, tracking of station utilization, worker guidance, on-the-fly quality control and many more

Developing the second net is the main subject of my work. The thesis includes everything from setting up an efficient training environment, write robust input pipelines and workflows, explore different architectures, simulate different hardware parameters and find a reliable, generalising solution.

Nets are trained on a simulation-PC with a RTX3080Ti & 16 CPU-cores, wandb is used for sweeping different architectures and input pipelines/operations, models are shared between different agents.

The models are constructed using tensorflow and include custom layers for preprocessing

The project is embedded in a larger code-environment that is not public. For this reason, code related to the thesis is publised here in parallel.

If one would like to redo the experiments, feel free to reach out to @BeeJayK for the training data.
One would need to: 
- clone the core-repository and the helpers-repository
- install the helpers-repository via pip
- fill in the data in the core-repository where it's needed
- possibly set up IT connections for training-PC and data synchronisation as shown in the thesis

The following provides an overview on the IT connections used for developement:
<img width="1244" alt="github-repo-it-connections" src="https://user-images.githubusercontent.com/75037677/162510194-474bff88-38c5-47fd-9762-10e9b4ea7880.png">

