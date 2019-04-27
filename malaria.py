# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:51:03 2019

@author: abhis
"""

import os
import glob

#directories
base_dir=os.path.join("D:/datasets/malaria/cell_images")
infected_dir=os.path.join(base_dir,"Parasitized")
normal_dir=os.path.join(base_dir,"Uninfected")

infected_files=glob.glob(infected_dir+"/*.png")
normal_files=glob.glob(normal_dir+"/*.png")
#dataset contains 13779 imagess each in a folder
#creating the dataframe

import numpy as np
import pandas as pd

np.random.seed(42)

file_df=pd.DataFrame({'filename':infected_files+normal_files,'labels':['malaria']*len(infected_files)+['normal']*len(normal_files)}).sample(frac=1,random_state=42).reset_index(drop=True)

