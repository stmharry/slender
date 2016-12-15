from slender.producer import LocalFileProducer as Producer
from slender.processor import TrainProcessor as Processor
from slender.net import TrainNet as Net
from slender.util import new_working_dir

from env import IMAGE_DIR, WORKING_DIR_ROOT

WORKING_DIR = new_working_dir(WORKING_DIR_ROOT)
BATCH_SIZE = 64
SUBSAMPLE_FN = Producer.SubsampleFunction.HASH(mod=64, divisible=False)
MIX_SCHEME = Producer.MixScheme.UNIFORM
GPU_FRAC = 0.6
LEARNING_RATE = 0.01
NUM_TRAIN_EPOCHS = 15
NUM_DECAY_EPOCHS = 1.5


class Factory(object):
    def __init__(self,
                 image_dir=IMAGE_DIR,
                 working_dir=WORKING_DIR,
                 batch_size=BATCH_SIZE,
                 subsample_fn=SUBSAMPLE_FN,
                 mix_scheme=MIX_SCHEME,
                 gpu_frac=GPU_FRAC,
                 learning_rate=LEARNING_RATE,
                 num_train_epochs=NUM_TRAIN_EPOCHS,
                 num_decay_epochs=NUM_DECAY_EPOCHS):

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
            learning_rate=learning_rate,
            learning_rate_decay_steps=num_decay_epochs * producer.num_batches_per_epoch,
            gpu_frac=gpu_frac,
        )
        blob = producer.blob().funcs([
            processor.preprocess,
            net.forward,
        ])
        self.__dict__.update(locals())

    def run(self):
        self.net.eval(self.num_train_epochs * self.producer.num_batches_per_epoch)


if __name__ == '__main__':
    factory = Factory()
    factory.run()
