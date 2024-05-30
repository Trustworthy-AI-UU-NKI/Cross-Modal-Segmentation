import matplotlib.pyplot as plt
import numpy as np

class Args:
    def __init__(self):
        self.pretrain = "xavier"
        self.weight_init = "xavier"
        self.layer = 8
        self.vc_num = 10
        self.true_clu_loss = True
        self.k2 = 10
        self.learning_rate = 0.0001