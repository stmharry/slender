from slender.producer import LocalFileProducer
from slender.processor import TestProcessor
from slender.net import TestNet
from slender.util import latest_working_dir

IMAGE_DIR = '/mnt/data/food-img'
WORKING_DIR = latest_working_dir('/mnt/data/food-save')

BATCH_SIZE = 4
SUBSAMPLE_FN = LocalFileProducer.hash_subsample_fn(64, divisible=True),
GPU_FRAC = 0.3


class Factory(object):
    def __init__(self,
                 image_dir=IMAGE_DIR,
                 working_dir=WORKING_DIR,
                 batch_size=BATCH_SIZE,
                 subsample_fn=SUBSAMPLE_FN,
                 gpu_frac=GPU_FRAC):

        self.producer = LocalFileProducer(
            image_dir=image_dir,
            working_dir=working_dir,
            is_training=False,
            batch_size=batch_size,
            subsample_fn=subsample_fn,
        )
        self.processor = TestProcessor()
        self.net = TestNet(
            working_dir=working_dir,
            num_classes=self.producer.num_classes,
            gpu_frac=gpu_frac,
        )
        self.blob = self.producer.blob().func(self.processor.preprocess).func(self.net.forward)

        self.__dict__.update(locals())

    def run(self):
        self.net.test(self.producer.num_batches_per_epoch)


if __name__ == '__main__':
    factory = Factory()
    factory.run()
