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

#spitting the datasets 
from sklearn.model_selection import train_test_split
from collections import Counter

train_files, test_files, train_labels, test_labels=train_test_split(file_df['filename'].values,file_df['labels'].values,test_size=0.3,random_state=42)
train_files, val_files, train_labels, val_labels=train_test_split(train_files,train_labels, test_size=0.1,random_state=42)
'''
print(train_files.shape,val_files.shape,test_files.shape)
print('Train:',Counter(train_labels),'\nVal:',Counter(val_labels),'\nTest:',Counter(test_labels))
'''
import cv2
from concurrent import futures
import threading

def get_img_shape_parallel(idx,img,total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs-1):
        print('{}:working on img num:{}'.format(threading.current_thread().name,idx))
    return cv2.imread(img).shape
ex=futures.ThreadPoolExecutor(max_workers=None)
data_inp=[(idx,img,len(train_files)) for idx,img in enumerate(train_files)]
print('starting Imgshape computation:')
train_img_dims_map=ex.map(get_img_shape_parallel, [record[0] for record in data_inp],[record[1] for record in data_inp],[record[2] for record in data_inp])
train_img_dims=list(train_img_dims_map)
print('Min Dimensions:',np.min(train_img_dims, axis=0))
print('Avg Dimesions:',np.mean(train_img_dims,axis=0))
print('Meedian Dimensions:',np.median(train_img_dims,axis=0))
print('Max dimensions:',np.max(train_img_dims,axis=0))
