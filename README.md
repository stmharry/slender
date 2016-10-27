# Slender: A Wrapper around TensorFlow-Slim for Easy Model Training and Serving

## Fast prototyping
```python
from slender.producer import LocalFileProducer
from slender.processor import TrainProcessor
from slender.net import TrainNet
from slender.util import new_working_dir

IMAGE_DIR = '/path/to/image/dir/that/contains/class names/as/subdirectories/'
WORKING_DIR = new_working_dir('/path/to/root/working/dir')

BATCH_SIZE = awesomeness_of_your_gpu
SUBSAMPLE_FN = LocalFileProducer.hash_subsample_fn(64, divisible=True),
GPU_FRAC = between_zero_and_one__but_leave_some_gpu_for_evaluation
NUM_TRAIN_EPOCHS = your_patience__make_sure_not_too_small
NUM_DECAY_EPOCHS = one_ish__make_sure_not_too_small_as_well

producer = LocalFileProducer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    is_training=True,
    batch_size=BATCH_SIZE,
    subsample_fn=SUBSAMPLE_FN,
)
processor = TrainProcessor()
net = TrainNet(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
    learning_rate_decay_steps=NUM_DECAY_EPOCHS * producer.num_batches_per_epoch,
    gpu_frac=GPU_FRAC,
)
blob = producer.blob().func(processor.preprocess).func(net.forward)

net.train(NUM_TRAIN_EPOCHS * producer.num_batches_per_epoch)
```

## Easy evaluation
```python
producer = LocalFileProducer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    is_training=False,
    batch_size=BATCH_SIZE,
    subsample_fn=SUBSAMPLE_FN,
)
processor = TestProcessor()
net = TestNet(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
    gpu_frac=GPU_FRAC,
)
blob = producer.blob().func(processor.preprocess).func(net.forward)

net.test(self.producer.num_batches_per_epoch)
```

## Image integrity checking
```python
producer = LocalFileProducer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
)

producer.check()
```
