import numpy as np
import os
import skimage.io
import tensorflow as tf
import time
import PIL.Image

''' Directory
'''
def new_working_dir(working_dir_root):
    working_dir = os.path.join(working_dir_root, time.strftime('%Y-%m-%d-%H%M%S'))
    os.makedirs(working_dir)
    return working_dir


def latest_working_dir(working_dir_root):
    working_dirs = [os.path.join(working_dir_root, dir_) for dir_ in os.listdir(working_dir_root)]
    working_dir = max(working_dirs, key=os.path.getmtime)
    return working_dir

''' Scope
'''
scope_join = os.path.join

''' Image
'''

def imread(file_name):
    with skimage.io.util.file_or_url_context(file_name) as file_name:
        return np.array(PIL.Image.open(file_name), dtype=np.float32)

''' Utility
'''
def identity(x):
    return x


''' DEBUG
'''
def LOG(value, name=None, fn=tf.identity):
    value = tf.Print(value, [fn(value)], '{}: '.format(name or value.__name__))
    return value
