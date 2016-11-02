import collections
import flask
import numpy as np

from slender.util import latest_working_dir
from online import Task, Factory

TOP_K = 6
FORMAT_STR = '{:.4f}'

app = flask.Flask(__name__)
app.config.update(
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=False,
)

food_factory = Factory(working_dir=latest_working_dir('/mnt/data/food-save'))


def eval(task):
    results = []

    task.eval(factory=food_factory)
    for prediction in task.outputs:
        indices = np.argsort(prediction)[:-(TOP_K + 1):-1]
        classname_probs = collections.OrderedDict([
            (food_factory.producer.class_names[index], FORMAT_STR.format(prediction[index]))
            for index in indices
        ])
        results.append({
            'status': 'ok',
            'classes': classname_probs,
        })

    print('Send <-[{:.4f} s]-> Receive <-[{:.4f} s]-> Eval <-[{:.4f} s]-> Respond'.format(
        task.put_timestamp - task.send_timestamp,
        task.get_timestamp - task.put_timestamp,
        task.done_timestamp - task.get_timestamp,
    ))

    return results


@app.route('/classify/food_type', methods=['POST'])
def classify_food():
    files = sorted(flask.request.files.items(), key=lambda (key, value): int(key))
    contents = [file[1].read() for file in files]
    task = Task(inputs=contents)
    task.send_timestamp = float(flask.request.headers['timestamp'])
    return flask.json.jsonify(eval(task))
