# Introduction
This repository is meant to give insights on the code, that I developed to solve the tasks of my master's thesis.

The thesis is done at the Digital Capibility Center (DCC) Aachen. The DCC is a cooperation between an institut of the RWTH Aachen and McKinsey, with the goal to bring dititalisation and AI into the production industry.

The work topic to be solved with the thesis is the classification of motion classes, performed by a worker, during the packaging process in a production line, based on a video stream.

The stream is processed by two neural networks, of which the first extracts the handcoordinates from images (using google's mediapipe) and the second takes these coordinate representations along with aditional information and predicts the current motion class.

Based on reliable motion class predictions (action recognition), one can easily develop toplevel routines, that can solve tasks which are currently time consuming, complicated and expensive. These tasks include (dynamic) cycle time calculations, tracking of station utilization, worker guidance, on-the-fly quality control and many more

Developing the second net is the main subject of my work. The thesis includes everything from setting up an efficient training environment, write robust input pipelines and workflows, explore different architectures, simulate different hardware parameters and find a reliable, generalising solution.

Nets are trained on a simulation-PC with a RTX3080Ti & 16 CPU-cores, wandb is used for sweeping different architectures and input pipelines/operations, models are shared between different agents.

The models are constructed using tensorflow and include custom layers for preprocessing

The project is embedded in a larger code-environment that is not public. For this reason, code related to the thesis is publised here in parallel.