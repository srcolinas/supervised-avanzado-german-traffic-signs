# -*- coding: utf-8 -*-
"""
Created on Wed 13/06/2017

This model consists of conv->pool->conv->pool->conv->pool->
dense->dense->dense. This model uses 200 images per class in
the training set.

@author: Sebastian R-Colina (srcolinas@gmail.com,
                             https://github.com/srcolinas)
"""
print('Importing modules...')
import os
from datetime import datetime

from helpers import get_files_path
from helpers import get_train_and_val_sets
from helpers import load_data
from helpers import Fetcher
from helpers import get_model_params
from helpers import restore_model_params

import tensorflow as tf
from tensorflow.contrib.layers import flatten


ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'german-traffic-signs')
TRAIN_DIR = os.path.join(DATA_DIR, 'training-set')
TEST_DIR = os.path.join(DATA_DIR, 'test-set')
SAVER_DIR = os.path.join(ROOT_DIR, "tf_logs", "saver")
os.makedirs(SAVER_DIR, exist_ok = True)
SAVER_FILE = os.path.join(SAVER_DIR, 'model')
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
LOG_DIR = "{}/run-{}/".format(os.path.join(ROOT_DIR, "tf_logs", "tb"), now)
os.makedirs(LOG_DIR, exist_ok = True)

assert os.path.exists(DATA_DIR)
assert os.path.exists(TRAIN_DIR)
assert os.path.exists(TEST_DIR)
assert os.path.exists(SAVER_DIR)
assert os.path.exists(LOG_DIR)

print('Loading data')
train_files = get_files_path(TRAIN_DIR, n = 200)# use 200 images per class

X_train, X_val, y_train, y_val = get_train_and_val_sets(train_files, val_size = 0.2)
del train_files

print("shape of training data: {}".format(X_train.shape))
print("shape of training labels: {}".format(y_train.shape))
print("shape of validation data: {}".format(X_val.shape))
print("shape of validation labels: {}".format(y_val.shape))

fetcher = Fetcher(X_train, y_train)
del X_train, y_train

    
n_ch = 3
height = 32
width = height
n_outputs = 43

print('building graph..')
graph0 = tf.Graph()
with graph0.as_default():
    
    is_training = tf.placeholder_with_default(False, shape = (), name = "is_training")
    
    X = tf.placeholder(dtype = tf.float32, shape = (None, height, width, n_ch), name = "X")
    
    y = tf.placeholder(dtype = tf.int32, shape = (None), name = "y")
        
    def model(input_data):
        with tf.name_scope('model'):           
            # Convolutional layer #1
            conv1 = tf.layers.conv2d(
                inputs = input_data,
                filters = 40,
                kernel_size = 7,
                padding = "same",
                activation = tf.nn.relu)
            # Pooling layer #1
            pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=2,
                strides=1)
            # Convolutional layer #2
            conv2 = tf.layers.conv2d(
                inputs = pool1,
                filters = 20,
                kernel_size = 5,
                padding = "same",
                activation = tf.nn.relu)
            # Pooling layer #2
            pool2 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=2,
                strides=1)
            # Convolutional layer #3
            conv3 = tf.layers.conv2d(
                inputs = pool2,
                filters = 10,
                kernel_size = 3,
                padding = "same",
                activation = tf.nn.relu)
            # Pooling layer #3
            pool3 = tf.layers.max_pooling2d(
                inputs=conv3,
                pool_size=2,
                strides=1)
            pool3_flat = flatten(pool3)
            drop1 = tf.layers.dropout(pool3_flat, rate = 0.3, training = is_training)
            # Dense layer #1
            dense1 = tf.layers.dense(inputs=drop1,
                                    units=1024,
                                    activation=tf.nn.relu,
                                    use_bias = True)
            drop4 = tf.layers.dropout(dense1, rate = 0.3, training = is_training)
            # Dense layer #1
            dense2 = tf.layers.dense(inputs=drop4,
                                    units=512,
                                    activation=tf.nn.relu,
                                    use_bias = True)
            drop5 = tf.layers.dropout(dense2, rate = 0.3, training = is_training)            
            # Logits layer 
            return tf.layers.dense(drop5, n_outputs, use_bias = True)
    
    
    logits = model(X)
    
    # Loss
    with tf.name_scope('Loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy)
    
    # Optimizer
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)
        
    sum_corrects = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, y, 1),tf.float32))
    
    # Predictions
    prob = tf.nn.softmax(logits, name = 'Probabilities')
    pred = tf.argmax(prob,1, name = 'Prediction')
          
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
 
file_writer = tf.summary.FileWriter(LOG_DIR, graph = graph0) #Visualize graph in tensorboard
file_writer.close() 
 
def get_accuracy(sum_op, X_, y_, splits = 8):
    """This functions computes accuracy based
        on a counter of right predictions. This
        way we can split data sets and use less
        RAM to calculate accuracies"""
    n = len(y_)
    k = int(n/splits)
    prev = 0
    s = 0
    for i in range(1, splits + 1):
        if i == splits:
            s += sum_op.eval(feed_dict={X: X_[prev:], y: y_[prev:]})
        else:
            s += sum_op.eval(feed_dict={X: X_[prev:i*k], y: y_[prev:i*k]})
            prev = i*k
    return s/n
 
n_epochs = 1000
batch_size = 50
n_batches = 20

best_acc_val = 0
delta_progress = 0.0001
checks_since_last_progress = 0
max_checks_without_progress = 10
best_model_params = None 
print('runing model..')
with tf.Session(graph=graph0) as sess:
    sess.run(init)
    print('Initialized')
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetcher.fetch(epoch, batch_index, batch_size, n_batches)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, is_training: True})
        if (epoch % 10 == 0):
            acc_train = get_accuracy(sum_corrects, X_batch, y_batch, splits = 1)
            acc_val = get_accuracy(sum_corrects, X_val, y_val, splits = 16)
            if acc_val > best_acc_val + delta_progress:
                best_acc_val = acc_val
                checks_since_last_progress = 0
                best_model_params = get_model_params()
            else:
                checks_since_last_progress += 1
            print("Epoch: {:4} || ".format(epoch) + 
				  "Train accuracy: {:4.4} ||".format(acc_train) + 
                  "Current validation accuracy: {:4.4} || ".format(acc_val) + 
                  "Best validation accuracy: {:4.4}".format(best_acc_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    if best_model_params:
        restore_model_params(best_model_params)
        save_path = saver.save(sess, SAVER_FILE)

del fetcher		
  
del X_val
del y_val

test_files = get_files_path(TEST_DIR)
X_test, y_test = load_data(test_files)

del test_files

with tf.Session(graph = graph0) as sess:
    restore_model_params(best_model_params)
    acc_test = get_accuracy(sum_corrects, X_test, y_test, splits = 15)
    print("Final accuracy on test set:", acc_test)
	

