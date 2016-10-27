import collections
import json
import locust
import random

FILENAME_LABELS = [
    ('http://files.recipetips.com/kitchen/images/refimages/bread/types/Italian_bread_500.jpg', 'bread'),
    ('http://appforhealth.com/wp-content/uploads/2014/09/coffee.jpg', 'coffee'),
    ('http://www-tc.pbs.org/food/files/2012/07/History-of-Ice-Cream-1.jpg', 'ice_cream'),
    ('https://65.media.tumblr.com/9f59f09887583de30c8d75887c727d52/tumblr_miwt2hmQX91qdisf0o1_500.jpg', 'pizza'),
    ('http://img.timeoutbeijing.com/201511/20151112095143307_large.jpg', 'pot'),
    ('http://blogs.discovermagazine.com/crux/files/2013/08/bowl-of-rice1.jpg', 'rice'),
    ('http://sunfestmarket.com/wp-content/uploads/2014/04/Seafood.jpg', 'seafood'),
    ('https://s-media-cache-ak0.pinimg.com/originals/f3/01/8e/f3018e0f32b7a7a15c2f69a7acbbcb1f.jpg', 'sake'),
]


class TaskSet(locust.TaskSet):
    @locust.task
    def post(self):
        num_filenames = random.randint(1, len(FILENAME_LABELS))
        filename_labels = random.sample(FILENAME_LABELS, num_filenames)
        (filenames, labels) = zip(*filename_labels)

        response = self.client.post('/classify/food_type', json={'images': filenames})
        items = json.loads(response.text, object_hook=collections.OrderedDict)
        for (item, filename, label) in zip(items, filenames, labels):
            classname_prob = sorted(item['classes'].items(), key=lambda x: x[1])[-1]
            print('{}->{}({})[{}]'.format(label, classname_prob[0], float(classname_prob[1]), filename))


class User(locust.HttpLocust):
    task_set = TaskSet
    min_wait = 5000
    max_wait = 20000
