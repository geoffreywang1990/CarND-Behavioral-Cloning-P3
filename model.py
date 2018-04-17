import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import convolutional,pooling
from 
lines = []
with open('../data/driving_log.csv') as inputF:
    reader = csv.reader(inputF)
    for line in reader:
        lines.append(line)


Cimages = []
Limages = []
Rimages = []
measurements = []
for line in lines:
    centerImg = "../data/" + line[0]
    Cimages.append(cv2.imread(centerImg))
    leftImg = "../data/" + line[1]
    Limages.append(cv2.imread(leftImg))
    rightImg = "../data/" + line[2]
    Rimages.append(cv2.imread(rightImg))
    steering = float(line[3])
    throttle = float(line[4])
    brake = float(line[5])
    speed = float(line[6])
    measurements.append([ steering,throttle,brake,speed])

n_classes = len(np.unique(y_train))
shape = (X_train[0].shape[0],X_train[0].shape[1],X_train[0].shape[2]);
model = Sequential()
model.add(Flatten(input_shape=shape))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
