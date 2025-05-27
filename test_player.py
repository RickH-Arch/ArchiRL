import platform
from envs.FineGridPark.fine_grid_park_env import FineGridPark
import time
import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(__file__))

SYSTEM = platform.system()

if SYSTEM == "Windows":
    import keyboard

file_path = './data/fineGrid_park/4街3_fineGrid.csv'

matrix = pd.read_csv(file_path,header=None).values
# 将matrix中所有float('nan')替换为0
for array in matrix:
    for i in range(len(array)):
        if type(array[i]) == float:
            if np.isnan(array[i]):
                array[i] = 0

config = {
        "matrix": matrix,
        "div_ind": 2,
        "vision_range": 7,
        "render_mode": "human"
    }

park = FineGridPark(config=config)