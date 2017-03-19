#!/usr/bin/env python

import gflags
import sys

from slender.producer import LocalFileProducer as Producer
from slender.processor import TestProcessor as Processor
from slender.net import HashNet, TestScheme
from slender.util import latest_working_dir

gflags.DEFINE_string('image_dir', None, 'Image directory')
gflags.DEFINE_string('working_dir_root', None, 'Root working directory')
gflags.DEFINE_integer('num_bits', 64, 'Number of bits')
gflags.DEFINE_integer('batch_size', 16, 'Batch size')
gflags.DEFINE_integer('subsample_ratio', 64, 'Training image subsample')
gflags.DEFINE_integer('interval', 300, 'Evaluation interval')
gflags.DEFINE_integer('timeout', 600, 'Timeout in seconds')
gflags.DEFINE_float('gpu_frac', 1.0, 'Fraction of GPU used')
FLAGS = gflags.FLAGS


class Net(HashNet, TestScheme):
    pass

if __name__ == '__main__':
    FLAGS(sys.argv)
    working_dir = latest_working_dir(FLAGS.working_dir_root)

    producer = Producer(
        image_dir=FLAGS.image_dir,
        working_dir=working_dir,
        batch_size=FLAGS.batch_size,
        subsample_fn=Producer.SubsampleFunction.HASH(mod=FLAGS.subsample_ratio, divisible=True),
        mix_scheme=Producer.MixScheme.NONE,
    )
    processor = Processor()
    net = Net(
        working_dir=working_dir,
        num_bits=FLAGS.num_bits,
        gpu_frac=FLAGS.gpu_frac,
    )

    blob = (
        producer.blob()
        .f(processor.preprocess)
        .f(net.build)
    )
    net.run(
        producer.num_batches_per_epoch,
        eval_interval_secs=FLAGS.interval,
        timeout=FLAGS.timeout,
    )
