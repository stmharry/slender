import collections
import tensorflow as tf
import flask
import flask.views
import numpy as np

from slender.producer import PlaceholderProducer as Producer
from slender.processor import TestProcessor
from slender.net import OnlineClasssifyNet as Net
from slender.model import SimpleTask as Task, BatchFactory
from slender.util import Timer

TOP_K = 6
PRECISION_STR = '{:.4f}'


class Processor(TestProcessor):
    def preprocess_single(self, content):
        content = tf.decode_base64(content)
        return super(Processor, self).preprocess_single(content)


class Factory(BatchFactory):
    def __init__(self,
                 working_dir,
                 queue_size,
                 batch_size,
                 net_dim,
                 shorter_dim,
                 gpu_frac,
                 timeout_fn):

        super(Factory, self).__init__(
            batch_size=batch_size,
            queue_size=queue_size,
            timeout_fn=timeout_fn,
        )

        self.producer = Producer(
            working_dir=working_dir,
            batch_size=batch_size,
        )
        self.processor = Processor(
            net_dim=net_dim,
            shorter_dim=shorter_dim,
            batch_size=batch_size,
        )
        self.net = Net(
            working_dir=working_dir,
            num_classes=producer.num_classes,
            gpu_frac=gpu_frac,
        )

        with self.net.graph.as_default():
            self.blob = (
                self.producer.blob()
                .f(self.processor.preprocess)
                .f(self.net.forward)
                .f(self.processor.postprocess)
            )
            self.net.init()

        self.start()

    def run_one(self, inputs):
        indices = [index for (index, input_) in enumerate(inputs) if len(input_['photoContent']) > 0]
        contents = [inputs[index]['photoContent'] for index in indices]

        with Timer(message='factory.blob.eval(size={:d})'.format(len(contents))):
            blob_val = self.blob.eval(
                sess=self.net.sess,
                feed_dict={self.producer.contents: contents},
            )

        outputs = []
        for (index, input_) in enumerate(inputs):
            if index in indices:
                status = 'ok'
                prediction = blob_val.predictions[index]
            else:
                status = 'error'
                prediction = [1.0] + [0.0] * (factory.producer.num_classes - 1)

            num_classes = np.argsort(prediction)[::-1][:TOP_K]
            class_names = collections.OrderedDict([
                (factory.producer.class_names[num_class], PRECISION_STR.format(prediction[num_class]))
                for num_class in num_classes
            ])
            outputs.append({
                'status': status,
                'photoName': input_['photoName'],
                'classes': class_names,
            })

        return outputs


class View(flask.views.View):
    methods = ['POST']

    def __init__(self, factory):
        self.factory = factory

    def dispatch_request(self):
        inputs = flask.request.get_json()
        task_id = flask.request.headers.get('task-id', None)

        with Timer(message='task({:d}).eval(size={:d})'.format(task.task_id, len(task.inputs))):
            task = Task(
                inputs=inputs,
                task_id=task_id and int(task_id),
            )
            outputs = task.eval(factory=self.factory)

        return flask.json.jsonify(outputs)


class App(flask.Flask):
    def __init__(self, import_name):
        super(App, self).__init__(import_name)

        self.config.update(
            JSON_SORT_KEYS=False,
            JSONIFY_PRETTYPRINT_REGULAR=False,
        )

    def add_route(self, url, factory):
        self.add_url_rule(
            url,
            view_func=View.as_view(url, factory=factory),
        )
