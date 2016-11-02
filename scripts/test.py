from slender.producer import LocalFileProducer as Producer
from slender.processor import TestProcessor as Processor
from slender.net import TestNet as Net
from slender.util import latest_working_dir

IMAGE_DIR = '/mnt/data/food-img'
WORKING_DIR = latest_working_dir('/mnt/data/food-save')

BATCH_SIZE = 4
SUBSAMPLE_FN = Producer.hash_subsample_fn(64, divisible=True),
GPU_FRAC = 0.3


class Factory(object):
    def __init__(self,
                 image_dir=IMAGE_DIR,
                 working_dir=WORKING_DIR,
                 batch_size=BATCH_SIZE,
                 subsample_fn=SUBSAMPLE_FN,
                 gpu_frac=GPU_FRAC):

        producer = Producer(
            image_dir=image_dir,
            working_dir=working_dir,
            batch_size=batch_size,
            subsample_fn=subsample_fn,
        )
        processor = Processor()
        net = Net(
            working_dir=working_dir,
            num_classes=producer.num_classes,
            gpu_frac=gpu_frac,
        )
        blob = producer.blob().funcs([
            processor.preprocess,
            net.forward,
        ])
        self.__dict__.update(locals())

    def run(self):
        self.net.test(self.producer.num_batches_per_epoch)


if __name__ == '__main__':
    factory = Factory()
    factory.run()
