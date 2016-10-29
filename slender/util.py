import cStringIO
import numpy as np
import os
import re
import requests
import tensorflow as tf
import time
import PIL.Image


URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')
SESSION = requests.Session()

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
def read(file_name):
    if URL_REGEX.match(file_name) is not None:
        r = SESSION.get(file_name)
        fp = cStringIO.StringIO(r.content)
    else:
        fp = open(file_name, 'rb')

    s = np.array(fp.read())
    fp.close()
    return s


''' Utility function
'''
def identity(x):
    return x


''' DEBUG
'''
def LOG(value, name=None, fn=tf.identity):
    value = tf.Print(value, [fn(value)], '{}: '.format(name or value.__name__))
    return value
