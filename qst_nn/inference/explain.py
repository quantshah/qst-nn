import numpy as np


from tf_explain.core import GradCAM


from qulearn.data.preprocess import normalize


from skimage import color
from skimage import io
import datetime

import cv2



def grad_cam_explanation(model, x, y, x_true, cutoff=0.9,
    heatmap_weight=0.1, image_weight=0.9):
    """
    Args:
        model (tf.model): A TensorFlow model with a predict function
        x (ndarray): Data to explain which is fed to the model (might be noisy)
        y (ndarray): Label
        x_true (ndarray): The true underlying data (without noise). Could be same as x
        cutoff (float, optional): Cutoff for the heatmap.
        heatmap_weight (float, optional): The weight of the heatmap in the overlay
        image_weight (float, optional): The weight of the image in the overlay
    
    Returns:
        grads (ndarray): Array of normalized gradient values
        heatmap_mask (ndarray[bool]): A mask of 0/1 according to the cutoff applied to
                                      the heatmap
        overlayed (ndarray): Array of heatmap overlayed on the image.
    """
    explainer = GradCAM()
    predicted_class = classifier.predict(x.reshape(-1, 32, 32, 1))
    yidx = np.argmax(predicted_class, 1)[0]
    
    grid = explainer.explain((x.reshape(-1, 32, 32, 1),y),
                             model, class_index=np.argmax(predicted_class))

    grads = color.rgb2gray(grid)
    # heatmap_img = cv2.applyColorMap(grid, cv2.COLORMAP_JET)
    heatmap_mask = grads > cutoff

    overlayed = cv2.addWeighted(heatmap_mask.astype(np.float32), heatmap_weight,
                                normalize(x_true.astype(np.float32).reshape(32, 32, 1)),
                                image_weight, 0)
    return grads, heatmap_mask, overlayed

