"""
Luke Jackson I-Tap 2017
"""

import numpy as np
import sys
import shutil
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping
from keras.models import Sequential
from keras import backend
from keras.constraints import maxnorm
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
import os
import argparse


# Flush the stdout and stderr after each epoch
class Flush(Callback):
    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.flush()
        sys.stderr.flush()


def create_model(dataset_name, out_dir,
                 minibatch_size,
                 max_norm,
                 initial_adam_learning_rate,
                 maximum_number_of_epochs,
                 early_stopping_patience):

    basename = os.path.basename(dataset_name)

    # delete/create, to remove old data in the same directory.
    if os.path.exists(os.path.join(out_dir, basename)):
        shutil.rmtree(os.path.join(out_dir, basename))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, basename)):
        os.makedirs(os.path.join(out_dir, basename))

    f = open(dataset_name, "rb")
    X_train = np.load(f)
    y_train = np.load(f)
    X_val = np.load(f)
    y_val = np.load(f)
    X_test = np.load(f)
    y_test = np.load(f)
    NUMBER_OF_CLASSES = np.load(f)
    f.close()

    # Reshape the samples array into the
    # form (number_of_samples, depth, height, width).
    # Since our input is grayscale we only use one channel (i.e. depth=1).
    X_train = np.reshape(
        X_train, (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    X_val = np.reshape(
        X_val, (X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))
    X_test = np.reshape(
        X_test, (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))

    SAMPLE_SHAPE = X_train[0].shape

    # Convert labels to one-hot representation
    y_train = np_utils.to_categorical(y_train, NUMBER_OF_CLASSES)
    y_val = np_utils.to_categorical(y_val, NUMBER_OF_CLASSES)
    y_test = np_utils.to_categorical(y_test, NUMBER_OF_CLASSES)

    depth = 30
    model = Sequential()
    model.add(Convolution2D(
        depth, 5, 5, border_mode='same',
        W_constraint=maxnorm(max_norm),
        init='he_normal',
        input_shape=(SAMPLE_SHAPE[0], SAMPLE_SHAPE[1], SAMPLE_SHAPE[2])))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Dropout(0.1))

    depth *= 2
    model.add(Convolution2D(
        depth, 5, 5, init='he_normal',
        border_mode='same', W_constraint=maxnorm(max_norm)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Dropout(0.2))

    depth *= 2
    model.add(Convolution2D(
        depth, 5, 5, init='he_normal',
        border_mode='same', W_constraint=maxnorm(max_norm)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(2000, init='he_normal', W_constraint=maxnorm(max_norm)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(2000, init='he_normal', W_constraint=maxnorm(max_norm)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(NUMBER_OF_CLASSES, W_constraint=maxnorm(max_norm)))
    model.add(Activation('softmax'))

    adam = Adam(lr=initial_adam_learning_rate)
    model.compile(
        loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # We will reinitialize the model with these
    # weights later, when we retrain the model
    # on full dataset after determining the stopping
    # times using the validation set.
    saved_initial_weights = model.get_weights()

    stopping_times = []
    for i in range(2):
        results = model.fit(
            X_train, y_train,
            batch_size=minibatch_size,
            nb_epoch=maximum_number_of_epochs,
            # We have already selected which model architecture and parameters
            # to use, so we don't discriminate between validation and test
            # sets, and combine them both into a validation set
            validation_data=(
                np.concatenate((X_val, X_test)), np.concatenate((y_val, y_test))),
            shuffle=True,
            verbose=2,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss', patience=early_stopping_patience,
                    verbose=2, mode='auto'),
                Flush()]
            )

        stopping_times.append(len(results.epoch))
        print("stopped after ", stopping_times[-1], "epochs")

        # Divide the learning rate by 10
        backend.set_value(adam.lr, 0.1 * backend.get_value(adam.lr))

    # Now we will retrain the model again keeping in mind the stopping times that
    # we got by the early stopping procedure
    adam = Adam(lr=initial_adam_learning_rate)
    model.compile(
        loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.set_weights(saved_initial_weights)

    for i in range(2):
        results = model.fit(
            np.concatenate((X_train, X_val, X_test)),
            np.concatenate((y_train, y_val, y_test)),
            batch_size=minibatch_size,
            nb_epoch=stopping_times[i],
            shuffle=True,
            verbose=2,
            callbacks=[Flush()]
            )

        # Divide the learning rate by 10
        backend.set_value(adam.lr, 0.1 * backend.get_value(adam.lr))

    # Save the model representation and weights to files
    model_in_json = model.to_json()
    f = open(os.path.join(out_dir, basename, 'model_architecture.json'), 'w')
    f.write(model_in_json)
    f.close()
    model.save_weights(os.path.join(out_dir, basename, 'model_weights.h5'), overwrite=True)

    shutil.copyfile(os.path.join(os.path.dirname(dataset_name), "class_lookup.txt"),
                    os.path.join(out_dir, basename, "class_lookup.txt"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, help='The path and name of the data set', required=True)
    parser.add_argument('-output_path', type=str, help='The path to output the model/graph to', required=True)
    # optional args
    parser.add_argument('--minibach_size', type=int,
                        help='todo',
                        default=100)
    parser.add_argument('--max_norm', type=int,
                        help='Max-norm constraint on weights',
                        default=4)
    parser.add_argument('--initial_adam_learning_rate', type=float,
                        help='...',
                        default=0.01)
    parser.add_argument('--maximum_number_of_epochs', type=int,
                        help='higher value is longer training time',
                        default=150)
    parser.add_argument('--early_stopping_patience', type=int,
                        help='higher value is longer training time',
                        default=10)

    return parser.parse_args(argv)

if __name__ == "__main__":
    arg = parse_arguments(sys.argv[1:])
    create_model(arg.dataset, arg.output_path,
                 minibatch_size=arg.minibach_size,
                 max_norm=arg.max_norm,
                 initial_adam_learning_rate=arg.initial_adam_learning_rate,
                 maximum_number_of_epochs=arg.maximum_number_of_epochs,
                 early_stopping_patience=arg.early_stopping_patience)
