# fullyLocalFL.py
# Pytorch+PySyft implementation of the Fully Local training algorithm in Federated Learning settings,
# @Author  : wwt
# @Date    : 2019-8-19

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import sys
import os
import random
import numpy as np
import syft as sy
import matplotlib.pyplot as plt
from learning_tasks import MLmodelReg, MLmodelCNN
from learning_tasks import MLmodelSVM
from learning_tasks import svmLoss
import utils


