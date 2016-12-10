import flask

from slender.processor import List
from slender.util import latest_working_dir

from util import Factory, classify_func

app = flask.Flask(__name__)
app.config.update(
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=False,
)

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

classify = classify_func(
    app=app,
    url='/classify/food_types',
    factory=factory,
)
