import tensorflow as tf

from .util import to_list


class BaseBlob(object):
    def func(self, f):
        return f(self)

    def eval(self, fetches, feed_dict=None):
        sess = tf.get_default_session()
        return sess.run(fetches, feed_dict=feed_dict)


class ImageLabelBlob(BaseBlob):
    def __init__(self, images, labels=None):
        images = to_list(images)
        if labels is None:
            labels = [tf.constant(-1, dtype=tf.int64) for _ in images]
        else:
            labels = to_list(labels)

        self.images = images
        self.labels = labels

    def as_tuple_list(self):
        return zip(self.images, self.labels)

    def eval(self):
        fetches = {value.name: value for value in self.images + self.labels}
        return super(ImageLabelBlob, self).eval(fetches)


class GenericBlob(BaseBlob):
    def __init__(self, values):
        values = to_list(values)
        self.values = values

    def eval(self):
        fetches = {value.name: value for value in self.values}
        return super(GenericBlob, self).eval(fetches)
