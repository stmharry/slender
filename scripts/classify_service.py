import base64
import collections
import flask
import numpy as np
import time

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


def manufacture(task, factory):
    results = []

    task.put_timestamp = time.time()
    task.eval(factory=factory)
    task.done_timestamp = time.time()

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
    jsons = flask.request.get_json()
    contents = [base64.standard_b64decode(json['photoContent']) for json in jsons]
    task = Task(inputs=contents)
    task.send_timestamp = float(flask.request.headers['timestamp'])

    results = manufacture(task, food_factory)
    return flask.json.jsonify(results)
