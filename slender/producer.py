from __future__ import print_function

import numpy as np
import os
import tensorflow as tf

from .blob import Blob
from .util import scope_join_fn

_ = scope_join_fn('producer')


class BaseProducer(object):
    BufferCapacity = 256

    @staticmethod
    def queue_join(values_list, dtypes=None, shapes=None, enqueue_many=False, name='queue_join'):
        dtypes = dtypes or [value.dtype for value in values_list[0]]
        shapes = shapes or [value.get_shape()[enqueue_many:] for value in values_list[0]]

        queue = tf.FIFOQueue(
            capacity=BaseProducer.BufferCapacity,
            dtypes=dtypes,
            shapes=shapes,
            name=name,
        )

        if enqueue_many:
            enqueue_fn = queue.enqueue_many
        else:
            enqueue_fn = queue.enqueue

        enqueue_list = [enqueue_fn(values) for values in values_list]
        queue_runner = tf.train.QueueRunner(queue, enqueue_list)
        tf.train.add_queue_runner(queue_runner)

        return queue


class ImageNetBaseProducer(BaseProducer):
    ClassNameFileName = 'class_names.txt'

    def __init__(self,
                 working_dir=None,
                 image_dir=None,
                 batch_size=64):

        self.working_dir = working_dir
        self.image_dir = image_dir
        self.batch_size = batch_size

        if working_dir is not None:
            self.classname_path = os.path.join(working_dir, ImageNetBaseProducer.ClassNameFileName)
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


class ImageNetFileProducer(ImageNetBaseProducer):
    class SubsampleFunction(object):
        @staticmethod
        def NoSubsample():
            def subsample(string):
                return True
            return subsample

        @staticmethod
        def Hash(mod, divisible):
            def subsample(string):
                return bool(hash(string) % mod) != divisible
            return subsample

    class MixScheme:
        NoScheme = 0
        Uniform = 1

    def __init__(self,
                 working_dir=None,
                 image_dir=None,
                 batch_size=64,
                 num_parallels=8,
                 subsample_fn=SubsampleFunction.NoSubsample(),
                 mix_scheme=MixScheme.NoScheme):

        super(ImageNetFileProducer, self).__init__(
            working_dir=working_dir,
            image_dir=image_dir,
            batch_size=batch_size,
        )

        self.filenames_by_subdir = {}
        for subdir_name in os.listdir(image_dir):
            self.filenames_by_subdir[subdir_name] = []

            subdir_path = os.path.join(image_dir, subdir_name)
            for (file_dir, _, file_names) in os.walk(subdir_path, followlinks=True):
                for file_name in file_names:
                    if file_name.startswith('.'):
                        continue
                    if not file_name.endswith('.jpg'):
                        continue
                    if not subsample_fn(file_name):
                        continue

                    self.filenames_by_subdir[subdir_name].append(os.path.join(file_dir, file_name))

            print('Subdir {:s} ({:d})'.format(subdir_name, len(self.filenames_by_subdir[subdir_name])))

        self.num_files = sum(map(len, self.filenames_by_subdir.values()))
        self.num_batches_per_epoch = self.num_files // self.batch_size
        self.num_parallels = num_parallels
        self.subsample_fn = subsample_fn
        self.mix_scheme = mix_scheme

    def blob(self):
        with tf.variable_scope(_(None)):
            if self.mix_scheme == LocalFileProducer.MixScheme.NoScheme:
                filename_labels = []
                for (subdir_name, file_names) in self.filenames_by_subdir.items():
                    if subdir_name in self.class_names:
                        label = np.flatnonzero(self.class_names == subdir_name)[0]
                    else:
                        label = -1

                    for file_name in file_names:
                        filename_labels.append((file_name, label))

                (file_names, labels) = zip(*filename_labels)

                labels = tf.convert_to_tensor(labels, dtype=tf.int64)
                file_names = tf.convert_to_tensor(file_names, dtype=tf.string)

            elif self.mix_scheme == LocalFileProducer.MixScheme.Uniform:
                assert set(self.filenames_by_subdir.keys()) == set(self.class_names)

                file_names = []
                for class_name in self.class_names:
                    file_names_ = self.filenames_by_subdir[class_name]
                    file_name_ = tf.train.string_input_producer(file_names_, name=class_name).dequeue()
                    file_names.append(file_name_)

                labels = tf.random_shuffle(tf.to_int64(tf.range(self.num_classes)))
                file_names = tf.gather(tf.stack(file_names), labels)

            filename_label_queue = BaseProducer.queue_join(
                [(file_names, labels)],
                enqueue_many=True,
            )

            filename_content_labels = []
            for num_parallel in xrange(self.num_parallels):
                (filename, label) = filename_label_queue.dequeue()
                filename_content_labels.append([
                    filename,
                    tf.read_file(filename),
                    label,
                ])

            filename_content_label_queue = BaseProducer.queue_join(filename_content_labels)
            (self.filenames, self.contents, self.labels) = filename_content_label_queue.dequeue_many(self.batch_size)

        return Blob(contents=self.contents, labels=self.labels)


class ImageNetPlaceholderProducer(ImageNetBaseProducer):
    def __init__(self,
                 working_dir=None,
                 batch_size=64):

        super(ImageNetPlaceholderProducer, self).__init__(
            working_dir=working_dir,
            batch_size=batch_size,
        )

    def blob(self):
        with tf.variable_scope(_(None)):
            self.contents = tf.placeholder(tf.string, shape=(None,))

            label_default = -1 * tf.ones_like(self.contents, dtype=tf.int64)
            self.labels = tf.placeholder_with_default(label_default, shape=(None,))

        return Blob(contents=self.contents, labels=self.labels)


LocalFileProducer = ImageNetFileProducer
PlaceholderProducer = ImageNetPlaceholderProducer
