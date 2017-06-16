# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 17:01:48 2017
This file stores several helper functions for the main scripts

@author: Sebastian
"""
import os
import random
import numpy as np
from skimage import io #Para leer las imágenes
from skimage import img_as_float #Para escalar las imágenes
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tqdm import tqdm 


def get_files_path(path, ext = ".jpg", n = np.inf):
    file_paths = []
    for folder in tqdm(os.listdir(path)):
        files = os.listdir(os.path.join(path,folder))
        if len(files) <= n:
            file_paths += [os.path.join(path,folder,file) for file in files if file.endswith(ext)]
        else:
            file_paths += [os.path.join(path,folder, file) for file in random.sample(files,n) if file.endswith(ext)]
            
    return file_paths
    
def get_files_path2(path, n_folders, n_imgs, ext = ".jpg",):
    file_paths = []
    for folder in tqdm(random.sample(os.listdir(path), n_folders)):
        files = os.listdir(os.path.join(path,folder))
        if len(files) <= n_imgs:
            file_paths += [os.path.join(path,folder,file) for file in files if file.endswith(ext)]
        else:
            file_paths += [os.path.join(path,folder, file) for file in random.sample(files,n_imgs) if file.endswith(ext)]
            
    return file_paths


def load_data(files, scale = True):
    X, y = [], []
    for file in tqdm(files):
        if file.endswith(".jpg"):
            y.append(int(os.path.split(os.path.split(file)[0])[-1]))
            if scale:
                X.append(img_as_float(io.imread(file)))
            else:
                X.append(io.imread(file))
            
    return np.array(X), np.array(y)

       
def get_train_and_val_sets(files, val_size = 0.2):
    X, y = load_data(files)
    return train_test_split(X, y, test_size=val_size, random_state=42)
    
def get_data_splits(X, y, splits):
    n = len(y)
    k = int(n/splits)
    prev = 0
    out = []
    for i in range(1, splits + 1):
        if i==splits:
            out.append((X[prev:],y[prev:]))
        else:
            out.append((X[prev:i*k],y[prev:i*k]))
        prev = i*k
    return out
    
class Fetcher:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.length = len(y)
        
    def fetch(self, epoch, batch_index, batch_size, n_batches):
        np.random.seed(epoch * n_batches + batch_index)          
        indices = np.unique(np.random.randint(self.length, size=batch_size))
        X_batch = self.X[indices,:,:,:]   
        y_batch = self.y[indices]   
        return X_batch, y_batch
		
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)
    
