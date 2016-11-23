from slender.producer import LocalFileProducer as Producer
from slender.processor import TestProcessor as Processor
from slender.net import TestNet as Net
from slender.util import latest_working_dir

from env import IMAGE_DIR, WORKING_DIR_ROOT

WORKING_DIR = latest_working_dir(WORKING_DIR_ROOT)
BATCH_SIZE = 16
SUBSAMPLE_FN = Producer.SubsampleFunction.HASH(mod=64, divisible=True)
MIX_SCHEME = Producer.MixScheme.NONE
GPU_FRAC = 0.3


class Factory(object):
    def __init__(self,
                 image_dir=IMAGE_DIR,
                 working_dir=WORKING_DIR,
                 batch_size=BATCH_SIZE,
                 subsample_fn=SUBSAMPLE_FN,
                 mix_scheme=MIX_SCHEME,
                 gpu_frac=GPU_FRAC):

        producer = Producer(
            image_dir=image_dir,
            working_dir=working_dir,
            batch_size=batch_size,
            subsample_fn=subsample_fn,
            mix_scheme=mix_scheme,
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
        self.net.eval(self.producer.num_batches_per_epoch)


if __name__ == '__main__':
    factory = Factory()
    factory.run()
