#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import h5py

#Constants
N_CLASSES=6
SHUFFLE_BUFFER=1000

def load_dataset():
    train_dataset = h5py.File('../datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('../datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#calculated values used as constants later


def train_eval_input_fn(features, labels, batch_size=1, num_epochs=1) :
  #preprocess input features and labels
  features["x"] = features["x"]/255.
  labels = convert_to_one_hot(labels,N_CLASSES).T

  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))

  # Shuffle, repeat and batch the examples
  dataset =dataset.shuffle(SHUFFLE_BUFFER).repeat(num_epochs).batch(batch_size)
  print(dataset)
  # Return the read end of the pipeline
  return dataset.make_one_shot_iterator().get_next()



def main(unused_argv) :
  X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
  train_eval_input_fn({"x" : X_train_orig},Y_train_orig,1)
  feature_columns=[tf.feature_column.numeric_column("x",shape=X_train_orig.shape)]

if __name__ == "__main__":
  tf.app.run()
