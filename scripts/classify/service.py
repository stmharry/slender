import base64
import collections
import flask

from slender.producer import PlaceholderProducer as Producer
from slender.processor import TestProcessor as Processor
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
            indices = sorted(range(len(output)), key=output.__getitem__)[::-1][:6]
            classname_probs = collections.OrderedDict([
                (factory.producer.class_names[index], '{:.4f}'.format(output[index]))
                for index in indices
            ])
            results.append({
                'status': 'ok',
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
        print('factory.blob.eval: batch_size={}'.format(len(inputs)))
        contents = [input_['photoRawContent'] for input_ in inputs]

        with Timer(message='factory.blob.eval'):
            blob_val = self.blob.eval(
                sess=self.net.sess,
                feed_dict={self.producer.contents: contents},
            )

        outputs = list(blob_val.predictions)
        return outputs


factory = Factory(
    working_dir=latest_working_dir('/mnt/data/food-save'),
    queue_size=1024,
    batch_size=16,
    net_dim=256,
    gpu_frac=1.0,
    timeout_fn=Factory.TimeoutFunction.QUARDRATIC(offset=0.02, delta=0.01),
)
factory.start()


@app.route('/classify/food_types', methods=['POST'])
def classify():
    json = flask.request.get_json()
    task_id = flask.request.headers.get('task-id', None)
    task_id = task_id and int(task_id)
    for item in json:
        item['photoRawContent'] = base64.standard_b64decode(item['photoContent'])

    task = Task(json, task_id=task_id)
    with Timer(message='{}.eval'.format(task)):
        results = task.eval(factory=factory)
    return flask.json.jsonify(results)
