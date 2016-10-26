from slender.producer import LocalFileProducer
from slender.processor import TrainProcessor
from slender.net import TrainNet
from slender.util import new_working_dir

IMAGE_DIR = '/mnt/data/food-img'
WORKING_DIR = new_working_dir('/mnt/data/food-save')

BATCH_SIZE = 64
SUBSAMPLE_FN = LocalFileProducer.hash_subsample_fn(64, divisible=False),
GPU_FRAC = 0.6
NUM_TRAIN_EPOCHS = 15
NUM_DECAY_EPOCHS = 1.5

class Factory(object):
    def __init__(self,
                 image_dir=IMAGE_DIR,
                 working_dir=WORKING_DIR,
                 batch_size=BATCH_SIZE,
                 subsample_fn=SUBSAMPLE_FN,
                 gpu_frac=GPU_FRAC,
                 num_train_epochs=NUM_TRAIN_EPOCHS,
                 num_decay_epochs=NUM_DECAY_EPOCHS):

        self.producer = LocalFileProducer(
            image_dir=image_dir,
            working_dir=working_dir,
            is_training=True,
            batch_size=batch_size,
            subsample_fn=subsample_fn,
        )
        self.processor = TrainProcessor()
        self.net = TrainNet(
            working_dir=working_dir,
            num_classes=producer.num_classes,
            learning_rate_decay_steps=num_decay_epochs * producer.num_batches_per_epoch,
            gpu_frac=gpu_frac,
        )
        self.blob = self.producer.blob().func(self.processor.preprocess).func(self.net.forward)

        self.__dict__.update(locals())

    def run(self):
        self.net.train(self.num_train_epochs * self.producer.num_batches_per_epoch)


if __name__ == '__main__':
    factory = Factory()
    factory.run()
