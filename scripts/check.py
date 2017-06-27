#!/usr/bin/env python

import gflags
import os
import sys
import tensorflow as tf

from slender.model import Task, BatchFactory

gflags.DEFINE_string('image_dir', None, 'Image directory')
gflags.DEFINE_integer('batch_size', 64, 'Batch size')
gflags.DEFINE_integer('num_parallels', 8, 'Number of parallel operations')

gflags.MarkFlagsAsRequired(['image_dir'])
FLAGS = gflags.FLAGS


class Factory(BatchFactory):
    def __init__(self,
                 batch_size,
                 num_parallels):

        super(Factory, self).__init__(
            batch_size=batch_size,
        )

        self.file_names = tf.placeholder(tf.string, shape=(None,))

        def check(file_name):
            content = tf.read_file(file_name)
            image = tf.image.decode_jpeg(content)
            shape = tf.shape(image)
            valid = tf.logical_and(tf.equal(tf.size(shape), 3), tf.equal(shape[2], 3))
            return valid

        self.valids = tf.map_fn(
            check,
            self.file_names,
            dtype=tf.bool,
            parallel_iterations=num_parallels,
        )
        self.sess = tf.Session()
        self.start()

    def run_one(self, inputs):
        sys.stderr.write('.')
        outputs = self.sess.run(
            self.valids,
            feed_dict={self.file_names: inputs},
        )
        return outputs


if __name__ == '__main__':
    FLAGS(sys.argv)

    factory = Factory(
        batch_size=FLAGS.batch_size,
        num_parallels=FLAGS.num_parallels,
    )

    subdir_names = os.listdir(FLAGS.image_dir)
    num_subdirs = len(subdir_names)
    for (num_subdir, subdir_name) in enumerate(subdir_names):
        subdir_path = os.path.join(FLAGS.image_dir, subdir_name)

        for (file_dir, _, file_names) in os.walk(subdir_path, followlinks=True):
            num_files = len(file_names)
            sys.stderr.write('Checking subdir {:s} ({:d}/{:d}), filedir {:s}, {:d} files\n'.format(
                subdir_name,
                num_subdir + 1,
                num_subdirs,
                file_dir,
                num_files,
            ))

            file_paths = []
            for (num_file, file_name) in enumerate(file_names):
                file_path = os.path.join(file_dir, file_name)
                file_paths.append(file_path)

            task = Task(inputs=file_paths)
            valids = task.eval(factory=factory)
            sys.stderr.write('\n')
            for (file_path, valid) in zip(file_paths, valids):
                if valid is None or not valid:
                    print('Exception raised on {:s}'.format(file_path))
                    os.remove(file_path)

    factory.stop()
