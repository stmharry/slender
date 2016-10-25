from foodnet.producer import LocalFileProducer
from foodnet.processor import TrainProcessor
from foodnet.net import TrainNet
from foodnet.util import new_working_dir

IMAGE_DIR = '/mnt/data/food-img'
WORKING_DIR = new_working_dir('/mnt/data/food-save')

BATCH_SIZE = 64
SUBSAMPLE = 64
GPU_FRAC = 0.6
NUM_STEPS = 50000
DECAY_EPOCHS = 1.5

producer = LocalFileProducer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    is_training=True,
    batch_size=BATCH_SIZE,
    subsample_fn=LocalFileProducer.hash_subsample_fn(SUBSAMPLE, divisible=False),
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
