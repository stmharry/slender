# Slender: A Wrapper around TensorFlow-Slim for Fast Model Training and Serving

## Fast prototyping
```python
from slender.producer import LocalFileProducer
from slender.processor import TrainProcessor
from slender.net import TrainNet

producer = LocalFileProducer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    is_training=True,
    batch_size=BATCH_SIZE,
)

processor = TrainProcessor()

net = TrainNet(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
    learning_rate_decay_steps=DECAY_EPOCHS * producer.num_files / producer.batch_size,
    gpu_frac=GPU_FRAC,
)

blob = producer.blob().func(processor.preprocess).func(net.forward).func(processor.postprocess)
net.train(NUM_STEPS)
```

## Image checking
```python
from slender.producer import LocalFileProducer

producer = LocalFileProducer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
)

producer.check()
```
