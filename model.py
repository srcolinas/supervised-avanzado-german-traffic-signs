# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 16:53:53 2017
This solution consist of applyin PCA to the data set
and then linear SVM to classify all images.
@author: Sebastian
"""

print('Importing modules ...')
import os

from helpers import get_files_path
from helpers import load_data
from helpers import Fetcher

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.externals import joblib

model = 'model_0'
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'german-traffic-signs')
TRAIN_DIR = os.path.join(DATA_DIR, 'training-set')
TEST_DIR = os.path.join(DATA_DIR, 'test-set')
SAVERS_DIR = os.path.join(ROOT_DIR, 'savers')
SAVER_DIR = os.path.join(SAVERS_DIR,model)
os.makedirs(SAVER_DIR, exist_ok = True)
SAVER_FILE = os.path.join(SAVER_DIR, 'model.pkl')


print('Loading training data ...')
train_files = get_files_path(TRAIN_DIR)
X, y = load_data(train_files)
X = np.reshape(X, newshape = (-1,32*32*3))
m = X.shape[0]

print('Doing dimentionality reduction ...')
ipca = IncrementalPCA(n_components = 500)
X_ipca = ipca.fit_transform(X)

clf = LinearSVC()
print('Fitting a linear SVM classfier ...')
clf.fit(X_ipca, y)
print('Score on training set',clf.score(X_ipca, y))

print('Loading test data ...')
test_files = get_files_path(TEST_DIR)
X, y = load_data(test_files)
X = np.reshape(X, newshape = (-1,32*32*3))
X_ipca = ipca.transform(X)
print('Score on test set',clf.score(X_ipca, y))