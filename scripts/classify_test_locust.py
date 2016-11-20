import locust
import os
import random
import time

from slender.util import Timer
from classify_test_images import get_images, images_to_json, json_to_classnames

images = get_images()

class TaskSet(locust.TaskSet):
    @locust.task
    def post(self):
        num_files = random.randint(5, 15)
        indices = random.sample(range(len(images)), num_files)

        this_images = map(images.__getitem__, indices)
        json = images_to_json(this_images)

        message = 'task_set({}).client.post'.format(id(self))
        print('{}: batch_size={}'.format(message, len(indices)))
        with Timer(message=message):
            response = self.client.post(
                '/classify/food_type',
                headers={'task-id': str(id(self))},
                json=json,
                catch_response=True,
            )

        with response:
            json = response.json()
            class_names = [image.class_name for image in this_images]
            class_names_ = json_to_classnames(json)

            if class_names != class_names_:
                print('original: {}'.format(class_names))
                print('predicted: {}'.format(class_names_))
                print('json: {}'.format(json))
                response.failure('Mismatch!')


class User(locust.HttpLocust):
    task_set = TaskSet
    min_wait = 500
    max_wait = 1500
