import numpy as np

from slender.producer import PlaceholderProducer
from slender.processor import TestProcessor
from slender.net import OnlineNet

BATCH_SIZE = 64
GPU_FRAC = 1.0


class Factory(object):
    def __init__(self,
                 working_dir,
                 batch_size=BATCH_SIZE,
                 gpu_frac=GPU_FRAC):

        self.producer = PlaceholderProducer(
            working_dir=working_dir,
        )
        self.processor = TestProcessor()
        self.net = OnlineNet(
            working_dir=working_dir,
            num_classes=self.producer.num_classes,
            gpu_frac=gpu_frac,
        )
        self.blob = self.producer.blob().func(self.processor.preprocess).func(self.net.forward).func(self.processor.postprocess).func(self.net.init)

        self.__dict__.update(locals())

    def run(self, file_names):
        num_files = len(file_names)
        file_name_batches = np.array_split(file_names, (num_files - 1) // self.batch_size + 1)

        predictions = []
        for file_name_batch in file_name_batches:
            blob_val = self.blob.eval(feed_dict={self.producer.file_name: file_name_batch})
            predictions.append(blob_val.predictions)

        return np.concatenate(predictions, 0)
