from __future__ import print_function

import numpy as np
import os
import PIL.Image
import sys
import tensorflow as tf

from .blob import Blob
from .util import read


class BaseProducer(object):
    _SCOPE = 'producer'
    _CLASSNAME_NAME = 'class_names.txt'
    _BUFFER_CAPACITY = 128

    @staticmethod
    def get_queue_enqueue(values, dtypes, shapes):
        queue = tf.FIFOQueue(
            capacity=BaseProducer._BUFFER_CAPACITY,
            dtypes=dtypes,
            shapes=shapes,
        )
        enqueue = queue.enqueue_many(values)
        queue_runner = tf.train.QueueRunner(queue, [enqueue])
        tf.train.add_queue_runner(queue_runner)

        return (queue, enqueue)

    def __init__(self, working_dir, image_dir=None):
        self.working_dir = working_dir
        self.classname_path = os.path.join(working_dir, BaseProducer._CLASSNAME_NAME)
        self.image_dir = image_dir

        if os.path.isfile(self.classname_path):
            self.class_names = np.loadtxt(self.classname_path, dtype=np.str)
        else:
            self.class_names = np.sort([
                class_name
                for class_name in os.listdir(image_dir)
                if os.path.isdir(os.path.join(image_dir, class_name))
            ])
            np.savetxt(self.classname_path, self.class_names, fmt='%s')
        self.num_classes = len(self.class_names)


class LocalFileProducer(BaseProducer):
    @staticmethod
    def hash_subsample_fn(mod, divisible):
        def hash_subsample(string):
            return bool(hash(string) % mod) != divisible
        return hash_subsample

    def __init__(self,
                 image_dir,
                 working_dir,
                 batch_size=1,
                 subsample_fn=lambda v: True):

        super(LocalFileProducer, self).__init__(
            working_dir=working_dir,
            image_dir=image_dir,
        )

        self.is_training = is_training
        self.batch_size = batch_size

        self.filenames_per_class = [[
            os.path.join(file_dir, file_name)
                for (file_dir, _, file_names) in os.walk(os.path.join(image_dir, class_name))
                for file_name in file_names
                if not file_name.startswith('.')
                if subsample_fn(file_name)
            ] for class_name in self.class_names
        ]
        self.num_files = sum(map(len, self.filenames_per_class))
        self.num_batches_per_epoch = self.num_files // self.batch_size

    def check(self):
        for (num_class, (class_name, file_names)) in enumerate(zip(self.class_names, self.filenames_per_class)):
            file_names_ = []
            for (num_file, file_name) in enumerate(file_names):
                print('Class {} ({}/{}), File {} ({}/{})'.format(
                    class_name,
                    num_class + 1,
                    len(self.class_names),
                    file_name,
                    num_file + 1,
                    len(file_names),
                ), end='\033[K\r')
                sys.stdout.flush()

                try:
                    image = np.array(PIL.Image.open(file_name), dtype=np.float32)
                    assert image.ndim == 3 and image.shape[2] == 3
                except Exception:
                    print('Exception raised on {}'.format(file_name), end='\033[K\n')
                    os.remove(file_name)
                else:
                    file_names_.append(file_name)

            print('')

            self.filenames_per_class[num_class] = file_names_

    def blob(self):
        with tf.variable_scope(BaseProducer._SCOPE):
            if self.is_training:
                filename_per_class = [
                    tf.train.string_input_producer(file_names).dequeue()
                    for file_names in self.filenames_per_class
                ]

                labels = tf.random_shuffle(tf.to_int64(tf.range(self.num_classes)))
                file_names = tf.gather(tf.pack(filename_per_class), labels)
            else:
                (file_names, labels) = zip(*[
                    (file_name, label)
                    for (label, file_names) in enumerate(self.filenames_per_class)
                    for file_name in file_names
                ])

            (filename_label_queue, _) = BaseProducer.get_queue_enqueue(
                [file_names, labels],
                dtypes=[tf.string, tf.int64],
                shapes=[(), ()],
            )

            (self.file_name, self.label) = filename_label_queue.dequeue_many(self.batch_size)

        return Blob(file_name=self.file_name, label=self.label)


class PlaceholderProducer(BaseProducer):
    def __init__(self, working_dir):
        super(PlaceholderProducer, self).__init__(
            working_dir=working_dir,
        )

    def blob(self):
        with tf.variable_scope(BaseProducer._SCOPE):
            self.file_name = tf.placeholder(tf.string, shape=(None,))
            label_default = -1 * tf.ones_like(self.file_name, dtype=tf.int64)
            self.label = tf.placeholder_with_default(label_default, shape=(None,))

        return Blob(file_name=self.file_name, label=self.label)
