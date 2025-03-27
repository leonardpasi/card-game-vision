"""
This module contains functions used exclusively by the module NN.py, which
was used to train our neural networks
"""


import gzip
import numpy as np
import os


def load_segmented_figures():
    """
    Loads the segmented figures, and returns a balanced dataset for training

    Returns
    -------
    fig_lbl : numpy array
        contains the labels
    fig_pic : numpy array
        contains the corresponding pictures

    """
    current_folder = os.path.dirname(os.path.realpath(__file__))
    segmented_img_path = os.path.join(current_folder, 'output_selim_notebook')

    digits_lbl = np.load(os.path.join(segmented_img_path, 'test_data_digits.npy'))
    digits_pic = np.load(os.path.join(segmented_img_path, 'train_data_digits.npy'))
    
    # LABEL CORRECTIONS:
    digits_lbl[209] = 'J'
    digits_lbl[332] = 'Q'
    digits_lbl[207] = 'None' # Maybe it's actually a 2 that wasn't rotated?
    
    fig_id = np.logical_or(np.logical_or(digits_lbl == 'J', digits_lbl == 'Q'),
                           digits_lbl == 'K')
    
    fig_lbl = digits_lbl[fig_id]
    fig_pic = digits_pic[fig_id]
    
    fig_lbl = np.delete(fig_lbl, 78) # Remove figure 78, a queen rotated by 90°
    fig_pic = np.delete(fig_pic, 78, axis=0)
    
    fig_lbl = np.delete(fig_lbl, 3) # Remove king 3, to have a balanced dataset
    fig_pic = np.delete(fig_pic, 3, axis=0)
    
    return fig_lbl, fig_pic



def load_MNIST():
    current_folder = os.path.dirname(os.path.realpath(__file__))
    MNIST_path = os.path.join(current_folder, 'MNIST_data')
    
    
    # In the MNIST_data folder we place the four .gz files, given for lab3. The 
    # folder won't be submitted (too heavy). The following functions are taken
    # from lab3.
    
    image_shape = (28, 28)
    train_set_size = 60000
    test_set_size = 10000
    
    train_images_path = os.path.join(MNIST_path, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(MNIST_path, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(MNIST_path, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(MNIST_path, 't10k-labels-idx1-ubyte.gz')
    
    train_images = extract_data(train_images_path, image_shape, train_set_size)
    test_images = extract_data(test_images_path, image_shape, test_set_size)
    train_labels = extract_labels(train_labels_path, train_set_size)
    test_labels = extract_labels(test_labels_path, test_set_size)
    
    return train_images, test_images, train_labels, test_labels



def extract_data(filename, image_shape, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data



def extract_labels(filename, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def convert_lbl_2int(lbl, Figures = True):
    lbl_int = np.empty((lbl.shape), dtype = int)
    
    if Figures:
        lbl_int[lbl == 'J'] = 10
        lbl_int[lbl == 'Q'] = 11
        lbl_int[lbl == 'K'] = 12
    else:
        lbl_int[lbl == 'H'] = 0
        lbl_int[lbl == 'D'] = 1
        lbl_int[lbl == 'S'] = 2
        lbl_int[lbl == 'C'] = 3
    return lbl_int