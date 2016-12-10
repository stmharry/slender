import base64
import collections
import flask

from slender.producer import PlaceholderProducer as Producer
from slender.processor import TestProcessor as Processor, List
from slender.net import SimpleNet as Net
from slender.model import BaseTask, BatchFactory
from slender.util import latest_working_dir, Timer

app = flask.Flask(__name__)
app.config.update(
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=False,
)


class Task(BaseTask):
    def eval(self, factory):
        super(Task, self).eval(factory)

        results = []
        for (input_, output) in zip(self.inputs, self.outputs):
            if output is None:
                status = 'error'
                output = [1.0] + [0.0] * (factory.producer.num_classes - 1)
            else:
                status = 'ok'

            indices = sorted(range(len(output)), key=output.__getitem__)[::-1][:6]
            classname_probs = collections.OrderedDict([
                (factory.producer.class_names[index], '{:.4f}'.format(output[index]))
                for index in indices
            ])

            results.append({
                'status': status,
                'photoName': input_['photoName'],
                'classes': classname_probs,
            })

        return results


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

        producer = Producer(
            working_dir=working_dir,
            batch_size=batch_size,
        )
        processor = Processor(
            net_dim=net_dim,
            shorter_dim=shorter_dim,
            batch_size=batch_size,
        )
        net = Net(
            working_dir=working_dir,
            num_classes=producer.num_classes,
            gpu_frac=gpu_frac,
        )
        blob = producer.blob().funcs([
            processor.preprocess,
            net.forward,
            processor.postprocess,
        ])
        net.init()
        self.__dict__.update(locals())

    def run_one(self, inputs):
        indices = [index for index in xrange(len(inputs)) if inputs[index]['photoContentDecoded'] is not None]
        contents = [inputs[index]['photoContentDecoded'] for index in indices]

        with Timer(message='factory.blob.eval(size={})'.format(len(contents))):
            blob_val = self.blob.eval(
                sess=self.net.sess,
                feed_dict={self.producer.contents: contents},
            )

        outputs = [None] * len(inputs)
        for (index, prediction) in zip(indices, blob_val.predictions):
            outputs[index] = prediction

        return outputs


factory = Factory(
    working_dir=latest_working_dir('/mnt/data/food-save'),
    queue_size=1024,
    batch_size=16,
    net_dim=256,
    shorter_dim=List([256, 512]),
    gpu_frac=0.3,
    timeout_fn=Factory.TimeoutFunction.QUARDRATIC(offset=0.02, delta=0.01),
)
factory.start()


@app.route('/classify/food_types', methods=['POST'])
def classify():
    items = flask.request.get_json()
    task_id = flask.request.headers.get('task-id', None)
    task_id = task_id and int(task_id)

    for item in items:
        try:
            content = base64.standard_b64decode(item['photoContent'])
            if len(content) == 0:
                content = None
        except:
            print('Exception raised by {}'.format(item['photoName']))
            content = None

        item['photoContentDecoded'] = content

    task = Task(items, task_id=task_id)
    with Timer(message='task({}).eval(size={})'.format(task.task_id, len(items))):
        results = task.eval(factory=factory)
    return flask.json.jsonify(results)
