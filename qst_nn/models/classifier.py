import tensorflow as tf
import tensorflow_addons as tfa


def Classifier():
    inp = tf.keras.layers.Input(shape=[32, 32, 1], name='input_image')
    x = tf.keras.layers.Conv2D(32, 3, strides=1,
                               use_bias=False,
                              )(inp)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(32, 3, strides=1,
                               use_bias=False,
                              )(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.GaussianNoise(0.005)(x)
    x = tf.keras.layers.Dropout(0.4)(x) 
    
    x = tf.keras.layers.Conv2D(32, 3, strides=2,
                               use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    

    x = tf.keras.layers.Conv2D(64, 3, strides=1,
                              use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.GaussianNoise(0.005)(x)
    x = tf.keras.layers.Dropout(0.4)(x)    
    
    x = tf.keras.layers.Conv2D(64, 3, strides=1,
                              use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(64, 3, strides=2,
                              use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dropout(0.4)(x)    
    
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dense(7)(x)

    return tf.keras.Model(inputs=inp, outputs=x)
