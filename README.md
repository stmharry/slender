# Slender: A Wrapper around TensorFlow-Slim for Easy Model Training and Serving

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
)
processor = Processor()
net = Net(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
    learning_rate_decay_steps=NUM_DECAY_EPOCHS * producer.num_batches_per_epoch,
    gpu_frac=GPU_FRAC,
)
blob = producer.blob().func(processor.preprocess).func(net.forward)

net.eval(NUM_TRAIN_EPOCHS * producer.num_batches_per_epoch)
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
    batch_size=BATCH_SIZE,
    subsample_fn=SUBSAMPLE_FN,
)
processor = Processor()
net = Net(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
    gpu_frac=GPU_FRAC,
)
blob = producer.blob().func(processor.preprocess).func(net.forward)

net.eval(producer.num_batches_per_epoch)
```

## Image integrity checking
```python
from slender.producer import LocalFileProducer as Producer

producer = Producer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
)

producer.check()
```

## Serving models online
```
Please refer to scripts/classify/service.py for now since I am too lazy ...
```
