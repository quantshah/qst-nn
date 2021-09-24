import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0002, 0.9, 0.9)
