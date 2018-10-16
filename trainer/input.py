import tensorflow as tf
import numpy as np

def train_fn():
    return input_fn(tf.estimator.ModeKeys.TRAIN)

def eval_fn():
    return input_fn(tf.estimator.ModeKeys.EVAL)

def preprocess(image, label=None):
    image = tf.expand_dims(image, -1)
    image = tf.cast(image, tf.float32)
    image = image/255 - .5
    if not label == None:
        return image, tf.cast(label, tf.int32)
    return image

def input_fn(mode=tf.estimator.ModeKeys.TRAIN):
    mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    (x_train, y_train), (x_test, y_test) = mnist
    if mode == tf.estimator.ModeKeys.EVAL:

        ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        ds = ds.batch(128).map(preprocess).prefetch(1).repeat(1)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds = ds.shuffle(1000).map(preprocess).batch(128).prefetch(1).repeat(None)
    else:
        raise ValueError(f'{mode} is not one of `TRAIN` or `EVAL`')
    iterator = ds.make_one_shot_iterator()
    image, label = iterator.get_next()
    return {'image':image}, label


def serving_receiver_fn():
    reciever_tensors = {
        # The size of input image is flexible.
        'image': tf.placeholder(tf.uint8, [None, None, None]),
    }

    # Convert give inputs to adjust to the model.
    features = {
        # Resize given images.
        #'image': tf.image.resize_images(
        #    reciever_tensors['image'], [28, 28]),
        'image': preprocess(reciever_tensors['image'])
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                    features=features)

