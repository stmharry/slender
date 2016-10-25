from foodnet.producer import LocalFileProducer
from foodnet.processor import TestProcessor
from foodnet.net import TestNet
from foodnet.util import latest_working_dir

IMAGE_DIR = '/mnt/data/food-img'
WORKING_DIR = latest_working_dir('/mnt/data/food-save')

BATCH_SIZE = 4
SUBSAMPLE = 64
GPU_FRAC = 0.3

producer = LocalFileProducer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    is_training=False,
    batch_size=BATCH_SIZE,
    subsample_fn=LocalFileProducer.hash_subsample_fn(SUBSAMPLE, divisible=True),
)

processor = TestProcessor()

net = TestNet(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
    gpu_frac=GPU_FRAC,
)

blob = producer.blob().func(processor.preprocess).func(net.forward).func(processor.postprocess)
net.test(num_steps=producer.num_files / producer.batch_size)
