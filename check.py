from foodnet.producer import LocalFileProducer
from foodnet.util import new_working_dir

IMAGE_DIR = '/mnt/data/food-img'
WORKING_DIR = new_working_dir('/mnt/data/food-save')

producer = LocalFileProducer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
)

producer.check()
