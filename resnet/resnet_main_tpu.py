#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
import absl.logging as _logging

import numpy as np
import tensorflow as tf
import h5py
from resnet_model import ResNet50Network
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.train import AdamOptimizer
import pprint

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
    'train_steps', default=112603,
    help=('The number of steps to use for training. Default is 112603 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

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
    'iterations_per_loop', default=100,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

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

def load_dataset(data_dir):
    train_dataset = h5py.File(data_dir+'/datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:],dtype=np.float32) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:],dtype=np.int32) # your train set labels

    test_dataset = h5py.File(data_dir+'/datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:],dtype=np.float32) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:],dtype=np.int32) # your test set labels

    classes = np.array(test_dataset["list_classes"][:],dtype=np.int32) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C,dtype=np.int32)[Y.reshape(-1)].T
    return Y

#calculated values used as constants later


def train_eval_input_fn(features, labels, batch_size=1, num_epochs=1) :
  #preprocess input features and labels
  features = features/255. 
  labels = convert_to_one_hot(labels,N_CLASSES).T

  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices((features,labels))

  # Shuffle, repeat and batch the examples
  dataset =dataset.shuffle(SHUFFLE_BUFFER).repeat(num_epochs).batch(batch_size)
  print(dataset)
  # Return the read end of the pipeline
  return dataset.make_one_shot_iterator().get_next()


def resnet_model_fn(features, labels, mode, params) :
  logits = ResNet50Network(features,N_CLASSES)
  print('logits',logits)

  ### EstimatorSpec for prediction mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'classes' : tf.argmax(logits, axis=1),
      'probabilities' : tf.nn.softmax(logits, name='softmax_tensor')
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
        'classify': tf.estimator.export.PredictOutput(predictions)
        })


  batch_size = params["batch_size"]
  #one_hot_labels =tf.one_hot(labels,N_CLASSES)
  loss = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  host_call=None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_global_step()

    optimizer = tf.train.AdamOptimizer()
    if FLAGS.use_tpu:
        # When using TPU, wrap the optimizer with CrossShardOptimizer which
        # handles synchronization details between different TPU cores. To the
        # user, this should look like regular synchronous training.
        optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

  else :
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits):
      """ 
        Evaluation metric function. Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

    eval_metrics = (metric_fn, [labels, logits])
  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics)



def main(unused_argv) :

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  config = tpu_config.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=max(600, FLAGS.iterations_per_loop),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores,
          per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long

  resnet_classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

 

  X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(FLAGS.data_dir)
  #feature_columns=[tf.feature_column.numeric_column("x",shape=X_train_orig.shape, normalizer_fn=lambda x: x/255.)]
  num_eval_images = X_test_orig.shape[0]

  if FLAGS.mode == 'eval':
    eval_steps = num_eval_images // FLAGS.eval_batch_size
    start_timestamp = time.time()  # This time will include compilation time
    eval_results = resnet_classifier.evaluate(
        input_fn=lambda params: train_eval_input_fn( X_test_orig,Y_test_orig,batch_size=params["batch_size"]),
        steps=eval_steps,
        #checkpoint_path=ckpt
        )
    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Eval results: %s. Elapsed seconds: %d' % (eval_results, elapsed_time))
  else :
    if FLAGS.mode == 'train':
      resnet_classifier.train(
          input_fn=lambda params : train_eval_input_fn( X_train_orig,
            Y_train_orig,
            batch_size=params["batch_size"],
            num_epochs=FLAGS.train_steps), 
          max_steps=FLAGS.train_steps)


  

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
