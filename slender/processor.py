import abc
import itertools
import math
import tensorflow as tf

from .blob import Blob
from .util import scope_join_fn

_ = scope_join_fn('processor')
identity = lambda x: x


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
        val_list = [
            tf.constant(val, dtype=self.dtype) if val is not None else None
            for val in self.val_list
        ]
        return val_list


class BaseProcessor(object):
    __metaclass__ = abc.ABCMeta

    _MEAN = [123.68, 116.78, 103.94]

    @staticmethod
    def _height_and_width(image):
        shape = tf.shape(image)
        return (shape[0], shape[1])

    @staticmethod
    def _set_shape(image, shape):
        image.set_shape(shape)
        return image

    @staticmethod
    def _decode_jpeg(content):
        image = tf.image.decode_jpeg(content, channels=3)
        image.set_shape((None, None, 3))
        image = tf.to_float(image)
        return image

    @staticmethod
    def _mean_subtraction(image):
        image = image - tf.constant(BaseProcessor._MEAN, dtype=tf.float32)
        return image

    @staticmethod
    def _resize(image, shorter_dim, aspect_ratio=None):
        if aspect_ratio is None:
            (original_height, original_width) = BaseProcessor._height_and_width(image)
            aspect_ratio = tf.cast(tf.truediv(original_width, original_height), tf.float32)

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
    def _rotate(image, angle):
        image = tf.contrib.image.rotate(image, angle)
        return image

    @staticmethod
    def _random_rotate(image):
        angle = tf.random_uniform((), maxval=2 * math.pi, dtype=tf.float32)
        return BaseProcessor._rotate(image, angle)

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
    def _apply(func, arg_list_dict):
        return [
            func(**dict(zip(arg_list_dict.keys(), arg_value)))
            for arg_value in itertools.product(*arg_list_dict.values())
        ]

    def __init__(self,
                 net_dim,
                 num_duplicates=1,
                 batch_size=64):

        self.net_dim = net_dim
        self.shape = (net_dim, net_dim, 3)
        self.num_duplicates = num_duplicates
        self.batch_size = batch_size

    def decode_jpeg(self, content):
        return [BaseProcessor._decode_jpeg(content)] * self.num_duplicates

    def set_shape(self, images):
        return BaseProcessor._apply(BaseProcessor._set_shape, {
            'image': images,
            'shape': [self.shape],
        })

    def mean_subtraction(self, images):
        return BaseProcessor._apply(BaseProcessor._mean_subtraction, {
            'image': images,
        })

    def resize(self, images, shorter_dim, aspect_ratio):
        return BaseProcessor._apply(BaseProcessor._resize, {
            'image': images,
            'shorter_dim': shorter_dim.get(),
            'aspect_ratio': aspect_ratio.get(),
        })

    def random_crop(self, images):
        return BaseProcessor._apply(BaseProcessor._random_crop, {
            'image': images,
            'crop_height': [self.net_dim],
            'crop_width': [self.net_dim],
        })

    def central_crop_or_pad(self, images):
        return BaseProcessor._apply(BaseProcessor._central_crop_or_pad, {
            'image': images,
            'target_height': [self.net_dim],
            'target_width': [self.net_dim],
        })

    def random_rotate(self, images):
        return BaseProcessor._apply(BaseProcessor._random_rotate, {
            'image': images,
        })

    def random_flip(self, images):
        return BaseProcessor._apply(BaseProcessor._random_flip, {
            'image': images,
        })

    def adjust(self, images, delta, contrast):
        return BaseProcessor._apply(BaseProcessor._adjust, {
            'image': images,
            'delta': delta.get(),
            'contrast': contrast.get(),
        })

    @abc.abstractmethod
    def preprocess_images(self, images):
        pass

    def preprocess_single(self, content):
        images = self.decode_jpeg(content)
        images = self.preprocess_images(images)
        images = tf.stack(images)
        return images

    def preprocess(self, blob):
        with tf.variable_scope(_('preprocess')):
            images = tf.map_fn(
                self.preprocess_single,
                blob['contents'],
                dtype=tf.float32,
                parallel_iterations=self.batch_size,
            )

            shape = images.get_shape().as_list()
            new_shape = [-1] + shape[2:]
            self.images = tf.reshape(images, new_shape)

            self.num_repeats = shape[1]
            self.labels = tf.reshape(tf.tile(tf.expand_dims(blob['labels'], 1), (1, self.num_repeats)), (-1,))

        return Blob(images=self.images, labels=self.labels)

    def postprocess(self, blob):
        with tf.variable_scope(_('postprocess')):
            blob_dict = {}
            for (key, value) in blob.items():
                shape = value.get_shape().as_list()
                new_shape = [-1, self.num_repeats] + shape[1:]
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
                 batch_size=64):

        super(TrainProcessor, self).__init__(
            net_dim=net_dim,
            batch_size=batch_size,
        )

        self.shorter_dim = shorter_dim
        self.aspect_ratio = aspect_ratio
        self.delta = delta
        self.contrast = contrast

    def preprocess_images(self, images):
        images = self.mean_subtraction(images)
        images = self.resize(images, shorter_dim=self.shorter_dim, aspect_ratio=self.aspect_ratio)
        images = self.random_crop(images)
        images = self.random_flip(images)
        images = self.adjust(images, delta=self.delta, contrast=self.contrast)
        return images


class TestProcessor(BaseProcessor):
    def __init__(self,
                 net_dim=256,
                 shorter_dim=List([256, 512]),
                 aspect_ratio=List([1.0]),
                 batch_size=16):

        super(TestProcessor, self).__init__(
            net_dim=net_dim,
            batch_size=batch_size,
        )

        self.shorter_dim = shorter_dim
        self.aspect_ratio = aspect_ratio

    def preprocess_images(self, images):
        images = self.mean_subtraction(images)
        images = self.resize(images, shorter_dim=self.shorter_dim, aspect_ratio=self.aspect_ratio)
        images = self.central_crop_or_pad(images)
        return images
