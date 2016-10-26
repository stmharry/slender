import collections
import flask
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from slender.util import latest_working_dir
from online import Factory

app = flask.Flask(__name__)
app.config.update(JSON_SORT_KEYS=False)

TOP_K = 6
FORMAT_STR = '{:.4f}'

FACTORY_FOOD = Factory(working_dir=latest_working_dir('/mnt/data/food-save'))
CLASS_NAMES_FOOD = FACTORY_FOOD.producer.class_names


@app.route('/classify/food_type', methods=['POST'])
def classify_food():
    file_names = flask.request.json.get('images', [])

    results = []
    for prediction in FACTORY_FOOD.run(file_names):
        indices = np.argsort(prediction)[:-(TOP_K + 1):-1]
        classname_probs = collections.OrderedDict([
            (CLASS_NAMES_FOOD[index], FORMAT_STR.format(prediction[index]))
            for index in indices
        ])
        results.append({
            'status': 'ok',
            'classes': classname_probs,
        })

    return flask.json.jsonify(results)
