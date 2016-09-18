import tensorflow as tf

def to_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

def DEBUG(value, name=None, func=None):
    if not IS_DEBUG:
        return value

    if name is None:
        name = value.name
    show = value
    if func is not None:
        show = func(show)
        name = '%s(%s)' % (func.__name__, name)
    return tf.Print(value, [show], '%s: ' % name)
