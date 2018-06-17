#edited from https://github.com/llSourcell/How_to_simulate_a_self_driving_car.git
import csv
import pandas as pd
import tensorflow as tf
import numpy as np #matrix math
from sklearn.model_selection import train_test_split #to split out training and testing data 
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import SGD 
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
# using multi gpu
#from multi_gpu import make_parallel
#from keras.utils.training_utils import multi_gpu_model
#helper class to define input shape and generate training images given image paths & steering angles
from utils import INPUT_SHAPE, batch_generator,degree2rad,rad2degree
#for command line arguments
import argparse
#for reading files
import os

#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)

def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
#    lines = []
#    with open(args.training_path + '/driving_log.csv') as inputF:
#        reader = csv.reader(inputF)
#        for line in reader:
#            lines.append(line)
#    X = []
#    y = []
#    for line in lines:
#        centerImg = "IMG/" + line[0].split('/')[-1]
#        leftImg =   "IMG/" + line[1].split('/')[-1]
#        rightImg =  "IMG/" + line[2].split('/')[-1]
#        steering = float(line[3])
#        #throttle = float(line[4])
#        #brake = float(line[5])
#        #speed = float(line[6])
#        X.append([centerImg, leftImg, rightImg])
#        #y.append([steering,speed])
#        y.append(steering)
#    #now we can split the data into a training (80), testing(20), and validation set
#    #thanks scikit learn
#    #print np.array(y)#.shape," " ,np.array(y).shape
#    X_train, X_valid, y_train, y_valid = train_test_split(np.array(X), np.array(y), test_size=args.test_size, random_state=0)
#    #print(y_train)


    data_df = pd.read_csv(os.path.join(os.getcwd(), args.training_path, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    return X_train, X_valid, y_train, y_valid


def build_train_model(args, X_train, X_valid, y_train, y_valid):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 2 (output,steering and throutle)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    #model.add(Lambda(degree2rad, output_shape=[1]))
    model.summary()

    """
    Train the model
    """
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss='mse', optimizer=SGD(lr=args.learning_rate,momentum =0.8, decay = 1e-6), metrics=['mae'])

    #Fits the model on data generated batch-by-batch by a Python generator.
    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    model.fit_generator(batch_generator(args.training_path, X_train, y_train, args.batch_size , True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        validation_data=batch_generator(args.training_path, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='training_path',     type=str,   default='../training_data_img')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=40)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=400)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-2)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(args)
    #build model
    #model = make_parallel(build_model(args),4)
    #model = multi_gpu_model(model,gpus = 4)
    #train model on data, it saves as model.h5 
    build_train_model( args, *data)


if __name__ == '__main__':
    main()

