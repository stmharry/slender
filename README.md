# Slender: A Wrapper around TensorFlow for Model Training and Serving

## Fast prototyping
```python
from slender.producer import LocalFileProducer as Producer
from slender.processor import TrainProcessor as Processor
from slender.net import TrainNet as Net
from slender.util import new_working_dir

IMAGE_DIR = '/path/to/image/dir/that/contains/class_names/as/subdirectories/'
WORKING_DIR = new_working_dir('/path/to/root/working/dir')

BATCH_SIZE = awesomeness_of_your_gpu
GPU_FRAC = between_zero_and_one__but_leave_some_gpu_for_evaluation
NUM_TRAIN_EPOCHS = your_patience__make_sure_not_too_small
NUM_DECAY_EPOCHS = one_ish__make_sure_not_too_small_as_well

producer = Producer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    batch_size=BATCH_SIZE,
    subsample_fn=TRAIN_SUBSAMPLE_FN,
)
processor = Processor()
net = Net(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
    learning_rate_decay_steps=NUM_DECAY_EPOCHS * producer.num_batches_per_epoch,
    gpu_frac=GPU_FRAC,
)
blob = producer.blob().f(processor.preprocess).f(net.build)

net.run(NUM_TRAIN_EPOCHS * producer.num_batches_per_epoch)
```

## Easy evaluation
```python
from slender.producer import LocalFileProducer as Producer
from slender.processor import TestProcessor as Processor
from slender.net import TestNet as Net
from slender.util import latest_working_dir

IMAGE_DIR = '/path/to/image/dir/that/contains/class_names/as/subdirectories/'
WORKING_DIR = latest_working_dir('/path/to/root/working/dir')

producer = Producer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    subsample_fn=TEST_SUBSAMPLE_FN,
)
processor = Processor()
net = Net(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
)
blob = producer.blob().f(processor.preprocess).f(net.build)

net.run(producer.num_batches_per_epoch)
```

## Serving models online with asynchronous workers
```
from slender.producer import PlaceholderProducer as Producer
from slender.processor import List, TestProcessor as Processor
from slender.net import ClassifyNet, OnlineScheme
from slender.model import BatchFactory

class Net(ClassifyNet, OnlineScheme):
    pass


class Factory(BatchFactory):
    def __init__(self)
        super(Factory, self).__init__()

        self.producer = Producer(
            working_dir=working_dir,
            batch_size=batch_size,
        )
        self.processor = Processor()
        self.net = Net(
            working_dir=working_dir,
            num_classes=self.producer.num_classes,
        )

        self.blob = (
            self.producer.blob()
            .f(self.processor.preprocess)
            .f(self.net.build)
            .f(self.processor.postprocess)
        )
        self.net.run()
        self.start()

    def run_one(self, inputs):
        return your_awesome_service(inputs)
```
