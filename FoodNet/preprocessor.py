import numpy as np
import tensorflow as tf

MEAN = np.load('mean.npy')

class BasePreprocessor(object):
    NET_DIM = 224

    @staticmethod
    def get_dynamic_shape(value):
        return tf.shape(value)

    @staticmethod
    def get_static_shape(value):
        return tuple(value.get_shape().as_list())

    @staticmethod
    def random(random_range, dtype=tf.float32, pre_func=tf.identity, post_func=tf.identity):
        return post_func(tf.random_uniform(
            (),
            pre_func(random_range[0]),
            pre_func(random_range[1]),
            dtype=dtype,
        ))

    @staticmethod
    def random_resize(value, dim_range, aspect_ratio_range):
        aspect_ratio = BasePreprocessor.random(
            aspect_ratio_range,
            pre_func=tf.log,
            post_func=tf.exp,
        )
        shorter_dim = BasePreprocessor.random(dim_range)

        size = tf.cond(
            tf.less(aspect_ratio, 1.0),
            lambda: (shorter_dim / aspect_ratio, shorter_dim),
            lambda: (shorter_dim, shorter_dim * aspect_ratio),
        )

        value = tf.expand_dims(value, 0)
        value = tf.image.resize_bilinear(value, tf.to_int32(tf.pack(size)))
        value = tf.squeeze(value, [0])
        return value

    @staticmethod
    def random_crop(value, dim):
        shape = BasePreprocessor.get_dynamic_shape(value)

        begin = (
            BasePreprocessor.random(0, shape[0] - dim + 1, dtype=tf.int32),
            BasePreprocessor.random(0, shape[1] - dim + 1, dtype=tf.int32),
            0,
        )
        size = (dim, dim, 3)

        value = tf.slice(value, tf.pack(begin), tf.pack(size))
        return value

    @staticmethod
    def random_flip(value):
        value = tf.image.random_flip_left_right(value)

        return value

    @staticmethod
    def random_adjust(value, max_delta=63, contrast_range=(0.5, 1.5)):
        value = tf.image.random_brightness(value, max_delta=max_delta)
        value = tf.image.random_contrast(value, lower=contrast_range[0], upper=contrast_range[1])

        return value


    def __init__(self,
                 num_test_crops,
                 train_dim_range,
                 test_dim_range,
                 train_aspect_ratio_range,
                 test_aspect_ratio_range,
                 net_dim,
                 mean_path=MEAN_PATH):

        self.num_test_crops = num_test_crops
        self.train_dim_range = train_dim_range
        self.test_dim_range = test_dim_range
        self.train_aspect_ratio_range = train_aspect_ratio_range
        self.test_aspect_ratio_range = test_aspect_ratio_range

        self.net_dim = net_dim
        self.shape = (net_dim, net_dim, 3)

    def _train(self, image):
        image = BasePreprocessor.random_resize(image, dim_range=self.train_dim_range, aspect_ratio_range=self.train_aspect_ratio_range)
        image = BasePreprocessor.random_crop(image, size=self.net_dim)
        image = BasePreprocessor.random_flip(image)
        image = BasePreprocessor.random_adjust(image)
        image = image - self.mean

        image.set_shape(self.shape)
        return image

    def train(self, images):
        return map(self._train, images)

    def _test_map(self, image):
        image = BasePreprocessor.random_resize(image, dim_range=self.test_dim_range, aspect_ratio_range=self.test_aspect_ratio_range)
        image = BasePreprocessor.random_crop(image, size=self.net_dim)
        image = BasePreprocessor.random_flip(image)
        image = image - self.mean

        return image

    def _test(self, image):
        image = tf.tile(tf.expand_dims(image, dim=0), multiples=(self.num_test_crops, 1, 1, 1))
        image = tf.map_fn(self._test_map, image)

        image.set_shape((self.num_test_crops,) + self.shape)
        return image

    def test(self, images):
        return map(self._test, images)
