from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#from .model import model_fn
#from . import input

from model import model_fn
import input

#from importlib import reload
#reload(input)

TRAIN_STEPS = 1000
MODEL_DIR = './checkpoints'
SAVE_DIR = './models/mnist/'


if __name__ == "__main__":
    tf.reset_default_graph()
    

    estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=MODEL_DIR)

    estimator.train(input.train_fn, max_steps=TRAIN_STEPS)
    estimator.evaluate(input.eval_fn)

    estimator.export_saved_model(SAVE_DIR, input.serving_receiver_fn)
