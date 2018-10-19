#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import flags
import absl.logging as _logging



import tensorflow as tf
from . import resnet_model 

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    'use_tpu', default=True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific flags
flags.DEFINE_string(
    'data_dir', default="../datasets/",
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))



flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {"train_and_eval", "train", "eval"}.')

flags.DEFINE_integer(
    'train_steps', default=33,
    help=('The number of steps to use for training. Default is 112603 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer(
    'train_batch_size', default=32, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=32, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'steps_per_eval', default=5000,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help=(
        'Maximum seconds between checkpoints before evaluation terminates.'))

flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop', default=33,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'save_checkpoints_steps', default=33,
    help=('Number of steps to run on TPU before saving model'
          ))


flags.DEFINE_integer(
    'num_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

# TODO(chrisying): remove this flag once --transpose_tpu_infeed flag is enabled
# by default for TPU
flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))



#Constants
N_CLASSES=6
SHUFFLE_BUFFER=1000
IMAGE_SIZE_H, IMAGE_SIZE_W = 64, 64


def dataset_parser(value) :
  keys_to_features = { 
    'x' : tf.FixedLenFeature([], tf.string, ''),
    'y' : tf.FixedLenFeature([], tf.int64, -1)
  }
  parsed = tf.parse_single_example(value, keys_to_features)
  image_bytes = tf.cast(tf.decode_raw(parsed['x'],tf.uint8),tf.float32)
  image_bytes = image_bytes/255.
  image = tf.reshape(image_bytes,shape=[IMAGE_SIZE_H*IMAGE_SIZE_W*3])
  #image = tf.reshape(image_bytes,shape=[IMAGE_SIZE_H,IMAGE_SIZE_W,3])
  label = tf.cast(parsed['y'],dtype=tf.int32)
  #label =tf.one_hot(label,N_CLASSES)
  return image , label


def train_eval_tfrecord_input_fn(filename,batch_size=1,num_epochs=1) :
  filenames = [filename]
  dataset = tf.data.TFRecordDataset(filenames)

  dataset = dataset.shuffle(SHUFFLE_BUFFER)

  dataset = dataset.map(dataset_parser).repeat(num_epochs).batch(batch_size, drop_remainder=True)
  print(dataset)
  return dataset

def model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()
    hidden = tf.layers.dense(features, 10, activation=tf.nn.relu)
    output = tf.layers.dense(hidden, 1)

    predictions = output
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        loss = tf.nn.l2_loss(predictions - labels)

        # define train_op
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)

        # wrapper to make the optimizer work with TPUs
        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=global_step)

    if params['use_tpu']:
        # TPU version of EstimatorSpec
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op
        )
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op
        )


def train_input_fn(params={}):
  # make some fake regression data

  # TPUEstimator passes params when calling input_fn
  batch_size = params.get('batch_size', 16)
  filename = params.get('filename')
  num_epochs = params.get('num_epochs', 1)
  dataset = train_eval_tfrecord_input_fn(filename,batch_size=batch_size,num_epochs=num_epochs)

  # TPUs need to know all dimensions when the graph is built
  # Datasets know the batch size only when the graph is run
  def set_shapes(features, labels):
      features_shape = features.get_shape().merge_with([batch_size, None])
      labels_shape = labels.get_shape().merge_with([batch_size])

      features.set_shape(features_shape)
      labels.set_shape(labels_shape)

      return features, labels

  dataset = dataset.map(set_shapes)
  dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

  return dataset


def main(unused_argv):
  # pass FLAGS as params so the model_fn can use
  # the TPU specific args
  params = vars(FLAGS)

  if FLAGS.use_tpu:
      # additional configs required for using TPUs
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu)
      tpu_config = tf.contrib.tpu.TPUConfig(
          num_shards=8, # using Cloud TPU v2-8
          iterations_per_loop=FLAGS.iterations_per_loop
      )

      # use the TPU version of RunConfig
      config = tf.contrib.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          model_dir=FLAGS.model_dir,
          tpu_config=tpu_config,
          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
          save_summary_steps=33
      )

      # TPUEstimator
      estimator = tf.contrib.tpu.TPUEstimator(
          model_fn=model_fn,
          config=config,
          params=params,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size, # FIXME
          export_to_tpu=False
      )
  else:
      config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)

      estimator = tf.estimator.Estimator(
          model_fn,
          config=config,
          params=params
      )

  estimator.train(lambda : train_input_fn({'filename' : FLAGS.data_dir+'/datasets/train_signs.tfrecord', 'batch_size': FLAGS.train_batch_size}) , max_steps=FLAGS.train_steps)

if __name__ == "__main__" :
  tf.logging.set_verbosity(tf.logging.NIFO)
  tf.app.run()
