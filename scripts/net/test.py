from slender.producer import LocalFileProducer as Producer
from slender.processor import TestProcessor as Processor
from slender.net import TestClassifyNet as Net
from slender.util import latest_working_dir

from env import image_dir, working_dir_root

working_dir = latest_working_dir(working_dir_root)


if __name__ == '__main__':
    producer = Producer(
        image_dir=image_dir,
        working_dir=working_dir,
        batch_size=16,
        subsample_fn=Producer.SubsampleFunction.HASH(mod=64, divisible=True),
        mix_scheme=Producer.MixScheme.NONE,
    )
    processor = Processor()
    net = Net(
        working_dir=working_dir,
        num_classes=producer.num_classes,
        gpu_frac=0.15,
    )

    blob = (
        producer.blob()
        .f(processor.preprocess)
        .f(net.forward)
    )
    net.run(producer.num_batches_per_epoch)
