#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:49:14 2019

@author: adam
"""

import os

# initialize the list of class label names
CLASSES = [ "Disordered", "Ordered"]

# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-7
MAX_LR = 1e-2
BATCH_SIZE = 20
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 32

# define the path to the output training history plot and cyclical
# learning rate plot
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])
