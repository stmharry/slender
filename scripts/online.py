import numpy as np

# DEBUG
import tensorflow as tf
from tensorflow.python.client import timeline

from slender.producer import PlaceholderProducer as Producer
from slender.processor import TestProcessor as Processor
from slender.net import OnlineNet as Net

BATCH_SIZE = 64
NET_DIM = 256
GPU_FRAC = 1.0


class Factory(object):
    def __init__(self,
                 working_dir,
                 batch_size=BATCH_SIZE,
                 net_dim=NET_DIM,
                 gpu_frac=GPU_FRAC):

        self.producer = Producer(
            working_dir=working_dir,
        )
        self.processor = Processor(
            net_dim=net_dim,
        )
        self.net = Net(
            working_dir=working_dir,
            num_classes=self.producer.num_classes,
            gpu_frac=gpu_frac,
        )
        self.blob = self.producer.blob().funcs([
            self.processor.preprocess,
            self.net.forward,
            self.processor.postprocess,
            self.net.init,
        ])
        self.__dict__.update(locals())

    def run(self, file_names):
        num_files = len(file_names)
        file_name_batches = np.array_split(file_names, (num_files - 1) // self.batch_size + 1)

        predictions = []
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # DEBUG
        run_metadata = tf.RunMetadata()  # DEBUG

        for file_name_batch in file_name_batches:
            blob_val = self.blob.eval(feed_dict={self.producer.file_name: file_name_batch}, options=options, run_metadata=run_metadata)
            predictions.append(blob_val.predictions)

        trace = timeline.Timeline(step_stats=run_metadata.step_stats)  # DEBUG
        with open('online1.json', 'w') as f:  # DEBUG
            f.write(trace.generate_chrome_trace_format())  # DEBUG

        return np.concatenate(predictions, 0)
