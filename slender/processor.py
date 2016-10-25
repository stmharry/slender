import abc
import os
import tensorflow as tf

from .blob import Blob
from .util import scope_join, identity, imread

_SCOPE = 'processor'
_MEAN = [123.68, 116.78, 103.94]

_ = lambda name: scope_join(_SCOPE, name)

class Range(object):
    def __init__(self, range_, num_duplicates=1, dtype=tf.float32, pre_fn=identity, post_fn=identity):
        self.minval = range_[0]
        self.maxval = range_[1]
        self.num_duplicates = num_duplicates
        self.dtype = dtype
        self.pre_fn = pre_fn
        self.post_fn = post_fn

    def get(self):
        val_list = [self.post_fn(tf.random_uniform(
            (),
            minval=self.pre_fn(self.minval),
            maxval=self.pre_fn(self.maxval),
            dtype=self.dtype,
        )) for num_crop in xrange(self.num_duplicates)]
        return val_list


class List(object):
    def __init__(self, val_list, dtype=tf.float32):
        self.val_list = val_list
        self.dtype = dtype

    def get(self):
        val_list = [tf.constant(val, dtype=self.dtype) for val in self.val_list]
        return val_list


class BaseProcessor(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def _height_and_width(image):
        shape = tf.shape(image)
        return (shape[0], shape[1])

    @staticmethod
    def _resize(image, shorter_dim, aspect_ratio=None):
        if aspect_ratio is None:
            (original_height, original_width) = BaseProcessor._height_and_width(image)

            aspect_ratio = tf.truediv(original_width, original_height)

        resize_dims = tf.cond(
            tf.less(aspect_ratio, 1.0),
            lambda: (shorter_dim / aspect_ratio, shorter_dim),
            lambda: (shorter_dim, shorter_dim * aspect_ratio),
        )
        image = tf.image.resize_images(
            image,
            tf.to_int32(tf.convert_to_tensor(resize_dims)),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        return image

    @staticmethod
    def _crop(image, offset_height, offset_width, crop_height, crop_width):
        offsets = tf.convert_to_tensor((offset_height, offset_width, 0))
        crop_shape = tf.convert_to_tensor((crop_height, crop_width, 3))

        image = tf.slice(image, offsets, crop_shape)
        return image

    @staticmethod
    def _pad(image, offset_height, offset_width, pad_height, pad_width):
        (original_height, original_width) = BaseProcessor._height_and_width(image)

        paddings = tf.convert_to_tensor((
            (offset_height, pad_height - offset_height - original_height),
            (offset_width, pad_width - offset_width - original_width),
            (0, 0)
        ))
        image = tf.pad(image, paddings)
        return image

    @staticmethod
    def _random_crop(image, crop_height, crop_width):
        (original_height, original_width) = BaseProcessor._height_and_width(image)

        max_offset_height = original_height - crop_height + 1
        max_offset_width = original_width - crop_width + 1
        offset_height = tf.random_uniform((), maxval=max_offset_height, dtype=tf.int32)
        offset_width = tf.random_uniform((), maxval=max_offset_width, dtype=tf.int32)
        return BaseProcessor._crop(image, offset_height, offset_width, crop_height, crop_width)

    @staticmethod
    def _central_crop_or_pad(image, target_height, target_width):
        (original_height, original_width) = BaseProcessor._height_and_width(image)

        diff_height = target_height - original_height
        diff_width = target_width - original_width

        image = BaseProcessor._crop(
            image,
            offset_height=tf.maximum(-diff_height / 2, 0),
            offset_width=tf.maximum(-diff_width / 2, 0),
            crop_height=tf.minimum(target_height, original_height),
            crop_width=tf.minimum(target_width, original_width),
        )
        image = BaseProcessor._pad(
            image,
            offset_height=tf.maximum(diff_height / 2, 0),
            offset_width=tf.maximum(diff_width / 2, 0),
            pad_height=target_height,
            pad_width=target_width,
        )
        image.set_shape((target_height, target_width, 3))
        return image

    @staticmethod
    def _random_flip(image):
        image = tf.image.random_flip_left_right(image)
        return image

    @staticmethod
    def _adjust(image, delta, contrast):
        image = tf.image.adjust_brightness(image, delta=delta)
        image = tf.image.adjust_contrast(image, contrast_factor=contrast)
        return image

    @staticmethod
    def _mean_addition(image):
        image = image + tf.constant(_MEAN, dtype=tf.float32)
        return image

    @staticmethod
    def _mean_subtraction(image):
        image = image - tf.constant(_MEAN, dtype=tf.float32)
        return image

    def __init__(self,
                 net_dim,
                 shorter_dim,
                 aspect_ratio,
                 delta,
                 contrast,
                 is_keep_aspect_ratio,
                 num_duplicates,
                 num_threads):

        self.net_dim = net_dim
        self.shape = (net_dim, net_dim, 3)
        self.shorter_dim = shorter_dim
        self.aspect_ratio = aspect_ratio
        self.delta = delta
        self.contrast = contrast
        self.is_keep_aspect_ratio = is_keep_aspect_ratio
        self.num_duplicates = num_duplicates
        self.num_threads = num_threads

    def filename_to_image(self, file_name):
        image = tf.py_func(imread, [file_name], tf.float32)
        image.set_shape(self.shape)
        return image

    def resize(self, images):
        images = [
            BaseProcessor._resize(image, shorter_dim=shorter_dim, aspect_ratio=aspect_ratio)
            for image in images
            for shorter_dim in self.shorter_dim.get()
            for aspect_ratio in (self.aspect_ratio.get() if not self.is_keep_aspect_ratio else [None])
        ]
        return images

    def random_crop(self, images):
        images = [
            BaseProcessor._random_crop(image, crop_height=self.net_dim, crop_width=self.net_dim)
            for image in images
        ]
        return images

    def central_crop_or_pad(self, images):
        images = [
            BaseProcessor._central_crop_or_pad(image, target_height=self.net_dim, target_width=self.net_dim)
            for image in images
        ]
        return images

    def random_flip(self, images):
        images = [
            BaseProcessor._random_flip(image)
            for image in images
        ]
        return images

    def adjust(self, images):
        images = [
            BaseProcessor._adjust(image, delta=delta, contrast=contrast)
            for image in images
            for delta in self.delta.get()
            for contrast in self.contrast.get()
        ]
        return images

    def mean_subtraction(self, images):
        images = [
            BaseProcessor._mean_subtraction(image)
            for image in images
        ]
        return images

    @abc.abstractmethod
    def preprocess_single(self, file_name):
        pass

    def preprocess(self, blob):
        with tf.variable_scope(_('preprocess')):
            image = tf.map_fn(
                self.preprocess_single,
                blob.file_name,
                dtype=tf.float32,
                parallel_iterations=self.num_threads,
            )

            shape = image.get_shape().as_list()
            new_shape = [-1] + shape[2:]
            self.image = tf.reshape(image, new_shape)

            self.num_repeats = shape[1]
            self.label = tf.reshape(tf.tile(tf.expand_dims(blob.label, 1), (1, self.num_repeats)), (-1,))

        return Blob(image=self.image, label=self.label)

    def postprocess(self, blob):
        with tf.variable_scope(_('postprocess')):
            blob_dict = {}
            for (key, value) in blob._dict.iteritems():
                shape = value.get_shape().as_list()
                new_shape = [shape[0] / self.num_repeats, self.num_repeats] + shape[1:]
                value = tf.reshape(value, new_shape)
                value = tf.reduce_mean(value, 1)
                blob_dict[key] = value
                setattr(self, key, value)

        return Blob(**blob_dict)


class TrainProcessor(BaseProcessor):
    def __init__(self,
                 net_dim=224,
                 shorter_dim=Range((256, 512)),
                 aspect_ratio=Range((0.5, 2.0)),
                 delta=Range((-64, 64)),
                 contrast=Range((0.5, 1.5)),
                 is_keep_aspect_ratio=False,
                 num_duplicates=1,
                 num_threads=8):

        super(TrainProcessor, self).__init__(
            net_dim=net_dim,
            shorter_dim=shorter_dim,
            aspect_ratio=aspect_ratio,
            delta=delta,
            contrast=contrast,
            is_keep_aspect_ratio=is_keep_aspect_ratio,
            num_duplicates=num_duplicates,
            num_threads=num_threads,
        )

    def preprocess_single(self, file_name):
        images = [self.filename_to_image(file_name)] * self.num_duplicates
        images = self.mean_subtraction(images)
        images = self.resize(images)
        images = self.random_crop(images)
        images = self.random_flip(images)
        images = self.adjust(images)
        image = tf.pack(images)
        return image


class TestProcessor(BaseProcessor):
    def __init__(self,
                 net_dim=None,
                 shorter_dim=List([256, 384, 512]),
                 aspect_ratio=List([1.0]),
                 delta=List([0]),
                 contrast=List([1.0]),
                 is_keep_aspect_ratio=False,
                 num_duplicates=1,
                 num_threads=8):

        super(TestProcessor, self).__init__(
            net_dim=net_dim or max(shorter_dim.val_list),
            shorter_dim=shorter_dim,
            aspect_ratio=aspect_ratio,
            delta=delta,
            contrast=contrast,
            is_keep_aspect_ratio=is_keep_aspect_ratio,
            num_duplicates=num_duplicates,
            num_threads=num_threads,
        )

    def preprocess_single(self, file_name):
        images = [self.filename_to_image(file_name)] * self.num_duplicates
        images = self.mean_subtraction(images)
        images = self.resize(images)
        images = self.central_crop_or_pad(images)
        image = tf.pack(images)
        return image