#!/usr/bin/env python

import gflags
import joblib
import numpy as np
import sys

from slender.blob import Blob
from slender.producer import LocalFileProducer as Producer
from slender.processor import TestProcessor as Processor
from slender.net import ClassifyNet, OnlineScheme

gflags.DEFINE_string('image_dir', None, 'Image directory')
gflags.DEFINE_string('working_dir', None, 'Working directory')
gflags.DEFINE_string('output_path', None, 'Path for output results')
gflags.DEFINE_integer('num_bits', 64, 'Number of bits')
gflags.DEFINE_integer('batch_size', 16, 'Batch size')
gflags.DEFINE_float('gpu_frac', 1.0, 'Fraction of GPU used')
FLAGS = gflags.FLAGS


class Net(ClassifyNet, OnlineScheme):
    pass

if __name__ == '__main__':
    FLAGS(sys.argv)
    working_dir = FLAGS.working_dir

    producer = Producer(
        image_dir=FLAGS.image_dir,
        working_dir=working_dir,
        batch_size=FLAGS.batch_size,
        mix_scheme=Producer.MixScheme.NONE,
    )
    processor = Processor()
    net = Net(
        working_dir=working_dir,
        num_classes=producer.num_classes,
        gpu_frac=FLAGS.gpu_frac,
    )
    producer.blob().f(processor.preprocess).f(net.build)

    net.run()

    blob = Blob(
        filenames=producer.filenames,
        feats=net.feats,
        labels=net.labels,
    )

    blob_vals = {}
    for num_batch in xrange(producer.num_batches_per_epoch):
        print('{:d} / {:d}'.format(num_batch + 1, producer.num_batches_per_epoch))
        blob_val = blob.eval(sess=net.sess)
        for key in blob_val.keys():
            if key in blob_vals:
                blob_vals[key].append(blob_val[key])
            else:
                blob_vals[key] = [blob_val[key]]

    for (key, value) in blob_vals.iteritems():
        blob_vals[key] = np.concatenate(value)

    joblib.dump(blob_vals, FLAGS.output_path)
