from slender.producer import LocalFileProducer as Producer
from slender.util import new_working_dir

IMAGE_DIR = '/mnt/data/food-img'
WORKING_DIR = new_working_dir('/mnt/data/food-save')

producer = Producer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
)

producer.check()
