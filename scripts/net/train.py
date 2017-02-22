from slender.producer import LocalFileProducer as Producer
from slender.processor import TrainProcessor as Processor
from slender.net import TrainClassifyNet as Net
from slender.util import new_working_dir

from env import image_dir, working_dir_root

working_dir = new_working_dir(working_dir_root)


if __name__ == '__main__':
    producer = Producer(
        image_dir=image_dir,
        working_dir=working_dir,
        batch_size=64,
        subsample_fn=Producer.SubsampleFunction.HASH(mod=64, divisible=False),
        mix_scheme=Producer.MixScheme.UNIFORM,
    )
    processor = Processor()
    net = Net(
        working_dir=working_dir,
        num_classes=producer.num_classes,
        learning_rate=0.1,
        learning_rate_decay_steps=1.5 * producer.num_batches_per_epoch,
        gpu_frac=0.5,
    )

    blob = (
        producer.blob()
        .f(processor.preprocess)
        .f(net.forward)
    )
    net.run(15 * producer.num_batches_per_epoch)
