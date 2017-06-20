# -*- coding: utf-8 -*-
"""
Created on Wed 13/06/2017

This model consists of 8 CNN of the form
conv->pool->dense->dense.
This model uses all images in the training set, 
regardless of it being unbalanced

@author: Sebastian R-Colina (srcolinas@gmail.com,
                             https://github.com/srcolinas)
"""
print('Training 8 shallow CNN')

print('Importing modules...')
import os
from datetime import datetime

from helpers import get_files_path
from helpers import get_train_and_val_sets
from helpers import load_data
from helpers import Fetcher
from helpers import get_model_params
from helpers import get_data_splits
from helpers import restore_model_params
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib.layers import flatten


ROOT_DIR = os.getcwd()
DATA_DIR = 'C:\\Users\\srodri16\\Desktop\\german-traffic-signs'
TRAIN_DIR = os.path.join(DATA_DIR, 'training-set')
TEST_DIR = os.path.join(DATA_DIR, 'test-set')
SAVER_DIR = os.path.join(ROOT_DIR, "logs", "saver")
os.makedirs(SAVER_DIR, exist_ok = True)
SAVER_FILE = os.path.join(SAVER_DIR, 'model')
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
LOG_DIR = "{}/run-{}/".format(os.path.join(ROOT_DIR, "logs", "tb"), now)
os.makedirs(LOG_DIR, exist_ok = True)

assert os.path.exists(DATA_DIR)
assert os.path.exists(TRAIN_DIR)
assert os.path.exists(TEST_DIR)
assert os.path.exists(SAVER_DIR)
assert os.path.exists(LOG_DIR)

n_models = 8

print('Loading data')
train_files = get_files_path(TRAIN_DIR) # Use all files  
X, y = load_data(train_files)
X, y = shuffle(X, y, random_state = 42)   
data = get_data_splits(X,y,n_models)
del train_files
del X, y

for X_, y_ in data:
    print('shape of X:',X_.shape,'shape of y:',y_.shape)
    # plt.hist(y_)
    # plt.show()

fetchers = [Fetcher(X_,y_) for X_, y_ in data]
fetchers.append(fetchers[0])

del data  
    
n_ch = 3
height = 32
width = height
n_outputs = 43

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
                filters = 20,
                kernel_size = 5,
                padding = "same",
                activation = tf.nn.relu)
            # Pooling layer #1
            pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=2,
                strides=2)
            # Dense layer #1
            pool1_flat = flatten(pool1)
            dense1 = tf.layers.dense(inputs=pool1_flat,
                                    units=1024,
                                    activation=tf.nn.relu,
                                    use_bias = True)
            drop1 = tf.layers.dropout(dense1, rate = 0.25, training = is_training)
            # Logits layer 
            return tf.layers.dense(drop1, n_outputs, use_bias = True)
    
    
    logits = [model(X) for i in range(n_models)]  

    logits_total = tf.add_n(logits)
        
    def get_training_op(labels, logits):
        with tf.name_scope("training_op"):            
            # Loss
            with tf.name_scope('Loss'):
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
                loss = tf.reduce_mean(xentropy)
            
            # Optimizer
            with tf.name_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer()
                return optimizer.minimize(loss)
    
    training_ops = [get_training_op(y, logit) for logit in logits]

    def get_sum(labels, logits):
        with tf.name_scope('Sum'):
            return tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, y, 1),tf.float32))

    sums = [get_sum(y, logit) for logit in logits]
       
    sum_total = get_sum(y, logits_total)     
    # Predictions
    probabilities = [tf.nn.softmax(logit) for logit in logits]
    prediction = [tf.argmax(prob,1) for prob in probabilities]
        
    probs_total = tf.nn.softmax(logits_total)
    pred_total = tf.argmax(probs_total,1)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
	
file_writer = tf.summary.FileWriter(LOG_DIR, graph = graph0) #Visualize graph in tensorboard
file_writer.close()

n_epochs = 1000
batch_size = 100
n_batches = 5

def get_accuracy(sum_op, X_, y_, splits = 8):
    """This functions computes accuracy based
        on a counter of right predictions. This
        way we can split data sets to use less
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

stoppers = [False for i in range(n_models)]    
best_acc_val = [0 for i in range(n_models)]
checks_since_last_progress = [0 for i in range(n_models)]
max_checks_without_progress = 10
best_model_params = None 
print('runing model...')
with tf.Session(graph=graph0) as sess:
    sess.run(init)
    print('Creating file writer')
    print('Initialized')
    for epoch in range(n_epochs + 1):
        for batch_index in range(n_batches):
            for i in range(n_models):
                if not stoppers[i]:
                    X_batch, y_batch = fetchers[i].fetch(epoch,
                                                        batch_index,
                                                        batch_size, n_batches)
                    sess.run(training_ops[i],
                                feed_dict={X: X_batch, y: y_batch, is_training: True})
            if (epoch % 10 == 0) and batch_index == 0:
                print("Epoch: {:4} || ".format(epoch))
                for i in range(n_models):              
                    acc = get_accuracy(sums[i], fetchers[i].X,
                                        fetchers[i].y, splits = 3)
                    print("Training accuracy of model {}: {:4.4} ||".format(i,acc))
                    acc = get_accuracy(sums[i], fetchers[i+1].X,
                                        fetchers[i+1].y, splits = 3)
                    print("Validation accuracy of model {}: {:4.4} ||".format(i,acc))
                    print("-------------------------------------------------")
                    if acc > best_acc_val[i]:
                        best_acc_val[i] = acc
                        checks_since_last_progress[i] = 0
                    else:
                        checks_since_last_progress[i] += 1
                    if checks_since_last_progress[i] > max_checks_without_progress:
                        print('Early stopping model ',i)
                        stoppers[i] = True
        if not False in stoppers:
            print("Early stopping all!")
            break
    model_params = get_model_params()
    if model_params:
        save_path = saver.save(sess, SAVER_FILE)
                
del fetchers		
del X_batch
del y_batch

test_files = get_files_path(TEST_DIR)
X_test, y_test = load_data(test_files)

del test_files

with tf.Session(graph = graph0) as sess:
    restore_model_params(model_params)
    for i in range(n_models):
        print('Accuracy of network',i,':',
              get_accuracy(sums[i], X_test, y_test, splits = 8))
    print('Accuracy of all', get_accuracy(sum_total, X_test, y_test, splits = 8))
            
