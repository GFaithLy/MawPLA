import numpy as np
import time
import torch
import random
from scipy.stats import qmc
from scipy.integrate import odeint
#from utils import *

class Config:
    def __init__(self):

        self.model_parameters = {
            'seed': 100,
            'prob_dim': 256,  
            'fc_map_dim': 128
        }

        #self.data_parameters = {
        #    'weight_morgan': 0.7,
        #    'weight_label': 0.3
        #}

        self.time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        # GPU
        self.device = torch.device("cuda:1")
        self.level = 8
        self.mode = "zero"
        self.wave = "coif2"
        self.lambda2 = 100
        self.pinn = 0
        self.fno = 0
        self.wno = 1
        self.max_smi_len = 256
        self.depth = 5
        self.embed_dim = 256