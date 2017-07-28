#!/usr/bin/env python3
# -*- coding:utf8 -*-

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import sys
from sklearn import preprocessing
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# load train data
train_data = []
train_lab = []
fake_label = []
CC = ['bird','crowded']


with open("train_randomlize.arff", "r") as file:
    line = file.readline()
    while line != '@data\n':
        line = file.readline()

    while True:
        line = file.readline()
        if not line:
            break
        else:
            temp = line.strip().split(',')
            fake_label.append(temp.pop())
            if len(temp) != 648:
                print("...Incorrect data...")
                sys.exit()
            train_data.append(list(map(float, temp)))

    for i in range(len(fake_label)):
        if fake_label[i] in CC:
            k = CC.index(fake_label[i])
            tmp = [0]*2
            tmp[k] = 1
            train_lab.append(tmp)

print('.......load train data successfully........')


# load test data
test_data = []
test_lab = []
fake_label = []
CC = ['bird','crowded']


with open("test_randomlize.arff", "r") as file:
    line = file.readline()
    while line != '@data\n':
        line = file.readline()

    while True:
        line = file.readline()
        if not line:
            break
        else:
            temp = line.strip().split(',')
            fake_label.append(temp.pop())
            if len(temp) != 648:
                print("...Incorrect data...")
                sys.exit()
            test_data.append(list(map(float, temp)))

    for i in range(len(fake_label)):
        if fake_label[i] in CC:
            k = CC.index(fake_label[i])
            tmp = [0]*2
            tmp[k] = 1
            test_lab.append(tmp)

print('.......load test data successfully........')


# Converted to ndarray
train_data = np.asarray(train_data)
train_lab = np.asarray(train_lab)
test_data = np.asarray(test_data)
test_lab = np.asarray(test_lab)


# Standardization
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
train_data = min_max_scaler.fit_transform(train_data)
test_data = min_max_scaler.fit_transform(test_data)
print('......Standardization Successfully.......')


train_data = train_data.reshape([-1, 27, 24, 1])
test_data = test_data.reshape([-1, 27, 24, 1])


# Building convolutional network(LeNet-5)
def LNet():
    Indata = input_data(shape=[None, 27, 24, 1], name='input')
    # convolutional layer1
    conv1 = conv_2d(Indata, 32, 3, activation='relu', regularizer="L2")
    # max_pool layer1
    pool1 = max_pool_2d(conv1, 2)
    # local Response Normalization1
    lrn1 = local_response_normalization(pool1)
    # convolutional layer2
    conv2 = conv_2d(lrn1, 64, 3, activation='relu', regularizer="L2")
    # max_pool layer2
    pool2 = max_pool_2d(conv2, 2)
    # local Response Normalization2
    lrn2 = local_response_normalization(pool2)
    # fully connected layer1
    fc1 = fully_connected(lrn2, 128, activation='tanh')
    dout1 = dropout(fc1, 0.5)
    # fully connected layer2
    fc2 = fully_connected(dout1, 256, activation='tanh')
    dout2 = dropout(fc2, 0.5)
    # fully connected layer3
    fc3 = fully_connected(dout2, 2, activation='softmax')
    sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=200)
    network = regression(fc3, optimizer=sgd, loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir='tmp/LeNet_logs')
    model.fit(train_data, train_lab, n_epoch=1, validation_set=({'input': test_data}, {'target': test_lab}), show_metric=True, batch_size=100, run_id='LeNet_cnn')

    # save model
    #model.save('Model/LeNet')


    # Prediction
    test_predict = model.predict(test_data)

    return test_predict


if __name__ == '__main__':
    test_predict = LNet()


















