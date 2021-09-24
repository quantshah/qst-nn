import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def add_noise(im, sigma = None):
    im = im/np.max(im)
    im = tf.keras.layers.GaussianNoise(sigma)(im, training=True)
    im = im.numpy()
    return im


def normalize(im):
    """
    Normalize an image by dividing by the max value.
    
    Args:
        im (array): Image array of shape (n, n, c)
    
    Returns:
        array: Normalized image by dividing by maximum value
    """
    return im/np.max(im)


def remove_data(xdata, ydata, label):
    """
    Removes the data for a given label
    
    Args:
        xdata (array): The array of x values
        ydata (array): The array of y values
        label (int/str): The integer or str label in ydata
    
    Returns:
        x, y (array): The new x and y data after removing label.
    """
    non_random_indices = np.argwhere(ydata != [label])[:, 0]
    x = xdata[non_random_indices]
    y = ydata[non_random_indices]        
    return x, y


def remap_labels(arr):
    """
    Remaps the label for y train to get rid of random state labels.
    
    Args:
        arr (array): An array of labels to remap
    
    Returns:
        arr (array): The array which is relabeled according to the dicts.
    """
    mapping = {"fock":0,
         "coherent":1,
         "thermal":2,
         "cat":3,
          "binomial":4,
          "num":5,
          "gkp":6}
    
    reverse_mapping = {
          0:"fock",
         1:"coherent",
         2:"thermal",
         3:"random",
         4:"cat",
         5:"binomial",
         6:"num",
         7:"gkp"
          }

    reverse_mapping_correct = {
              0:"fock",
             1:"coherent",
             2:"thermal",
             3:"cat",
             4:"binomial",
             5:"num",
             6:"gkp"
              }
    return np.array([[mapping[reverse_mapping[arr[i][0]]]] for i in range(len(arr))]).reshape(arr.shape)



