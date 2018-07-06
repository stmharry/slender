import os
import tensorflow as tf
import time


def new_working_dir(working_dir_root):
    working_dir = os.path.join(working_dir_root, time.strftime('%Y-%m-%d-%H%M%S'))
    os.makedirs(working_dir)
    return working_dir


def latest_working_dir(working_dir_root):
    working_dirs = [os.path.join(working_dir_root, dir_) for dir_ in os.listdir(working_dir_root)]
    working_dir = max(working_dirs, key=os.path.getmtime)
    return working_dir


def scope_join_fn(scope):
    def scope_join(*args):
        return '/'.join(filter(None, [scope] + list(args)))
    return scope_join


def LOG(value, name=None, fn=tf.identity):
    value = tf.Print(value, [fn(value)], '{}: '.format(name or value.__name__))
    return value


class Timer(object):
    @staticmethod
    def time(message):
        def _time(func):
            def _func(*args, **kwargs):
                with Timer(message=message):
                    return func(*args, **kwargs)
            return _func
        return _time

    def __init__(self, message=''):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        print(u'{:s} --->'.format(self.message))
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(u'---> {:s}: interval={:.4f} s'.format(self.message, self.interval))
