import collections
import flask
import numpy as np

from slender.util import latest_working_dir
from online import Factory

app = flask.Flask(__name__)
app.config.update(
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=False,
)

TOP_K = 6
FORMAT_STR = '{:.4f}'

FOOD_FACTORY = Factory(working_dir=latest_working_dir('/mnt/data/food-save'))


@app.route('/classify/food_type', methods=['POST'])
def classify_food():
    file_names = flask.request.get_json().get('images', [])

    results = []
    for prediction in FOOD_FACTORY.run(file_names):
        indices = np.argsort(prediction)[:-(TOP_K + 1):-1]
        classname_probs = collections.OrderedDict([
            (FOOD_FACTORY.producer.class_names[index], FORMAT_STR.format(prediction[index]))
            for index in indices
        ])
        results.append({
            'status': 'ok',
            'classes': classname_probs,
        })

    return flask.json.jsonify(results)
