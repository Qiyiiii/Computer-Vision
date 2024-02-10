# CSC320 Fall 2022
# Assignment 1
# (c) Kyros Kutulakos, Towaki Takikawa, Esther Lin
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

import os
import numpy as np
from PIL import Image

def read_image(path):
    """Loads an image as a Numpy array. 

    Args:
        path (str): Path to the image. 

    Returns:
        (np.ndarray): [H, W, 4] image. Normalized to [0.0, 1.0].
    """
    if not os.path.exists(path):
        raise Exception(f"The path {path} does not exist!")
    image = np.array(Image.open(path))
    image = image.astype(np.float32) / 255.0

    # If the image contains no alpha channel, concatenate an alpha channel.
    if image.shape[-1] == 3:
        image = np.concatenate([image, np.ones_like(image[...,0:1])], axis=-1)
    return image

def write_image(image, path):
    """Loads an image as a Numpy array. 

    Args:
        image (np.ndarray): [H, W, (3 or 4)] image. 
        path (str): Path to save the image to.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    Image.fromarray((image * 255.0).astype(np.uint8)).save(path)

def create_coordinates(h, w):
    """Creates an image of x, y coordinates.

    The coordinates are in normalized coordinate space from [-1, 1].

    TODO(ttakikawa): Which way is up?

    Args:
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        (np.ndarray): [H, W, 2] image of 2D coordinates.
    """
    window_x = np.linspace(-1, 1, num=w)
    window_y = np.linspace(1, -1, num=h)
    coords = np.stack(np.meshgrid(window_x, window_y), axis=-1)
    return coords

def normalize_coordinates(coords, h, w):
    """Normalize a coords array with height and width.

    Args:
        coords (np.ndarray): Coordinate tensor of shape [N, 2] in integer pixel coordinate space.
        h (int): The height of the image.
        w (int): The width of the image.

    Returns:
        (np.ndarray): Coordiante tensor of shape [N, 2] in normalized [-1, 1] space.
    """
    coords_ = coords.copy().astype(np.float32)
    coords_[..., 0] = (coords_[..., 0] / w) * 2.0 - 1.0
    coords_[..., 1] = (coords_[..., 1] / h) * 2.0 - 1.0
    return coords_
