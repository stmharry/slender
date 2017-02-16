from slender.producer import LocalFileProducer as Producer
from slender.processor import TrainProcessor as Processor
from slender.net import ClassifyNet, TrainMixin
from slender.util import new_working_dir

from env import image_dir, working_dir_root

working_dir = new_working_dir(working_dir_root)
batch_size = 64
subsample_fn = Producer.SubsampleFunction.HASH(mod=64, divisible=False)
mix_scheme = Producer.MixScheme.UNIFORM
gpu_frac = 0.6
learning_rate = 0.1
num_train_epochs = 15
num_decay_epochs = 1.5


class Net(ClassifyNet, TrainMixin):
    pass


if __name__ == '__main__':
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
        net.eval,
    ])

    net.run(num_train_epochs * producer.num_batches_per_epoch)
