import h5py
import numpy as np
import pandas as pd
import tensorflow as tf


def convert_to_tfrecord(h5_file,tf_filename,cols,dtypes):
  f = h5py.File(h5_file,"r")
  npcol = {}
  for i,col in enumerate(cols) :
    npcol[col] = np.array(f[col][:],dtype=dtypes[i])
  print(npcol[cols[0]][0][0][0])
  writer = tf.python_io.TFRecordWriter(tf_filename)

  for i in range(len(npcol[cols[0]])) :
    feature = {"x" : _bytes_feature(tf.compat.as_bytes(npcol[cols[0]][i].reshape(-1).tostring())),
      "y" :_int64_feature(npcol[cols[1]][i].reshape(-1))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
  writer.close()



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == "__main__":
  convert_to_tfrecord("datasets/test_signs.h5","datasets/test_signs.tfrecord",["test_set_x","test_set_y"],[np.uint8,np.int32])
  convert_to_tfrecord("datasets/train_signs.h5","datasets/train_signs.tfrecord",["train_set_x","train_set_y"],[np.uint8,np.int32])


