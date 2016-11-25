from slender.producer import LocalFileProducer as Producer
from slender.util import new_working_dir

from env import IMAGE_DIR, WORKING_DIR_ROOT

WORKING_DIR = new_working_dir(WORKING_DIR_ROOT)

producer = Producer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
)

producer.check()
