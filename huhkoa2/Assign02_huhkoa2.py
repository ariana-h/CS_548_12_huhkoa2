import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os

PATH_TO_DATA= "/data/facade_dataset"

def read_input_and_real_image(index: int, is_test=False):
    if is_test:
        path_real = f"testA/{index}.jpg"
        path_input = f"testB/{index}.jpg"
    else:
        path_real = f"trainA/{index}_A.jpg"
        path_input = f"trainB/{index}_B.jpg"
    train_real_path = os.path.join(PATH_TO_DATA, path_real)
    train_input_path = os.path.join(PATH_TO_DATA, path_input)
    
    train_real = tf.io.read_file(train_real_path)
    train_real = tf.io.decode_jpeg(train_real)
    
    train_input = tf.io.read_file(train_input_path)
    train_input = tf.io.decode_jpeg(train_input)
    return train_input, train_real
