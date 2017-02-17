from slender.processor import List

from model import App, Factory

app = App(__name__)

'''
app.add_route(
    url='/classify/6_categories',
    factory=Factory(
        working_dir='/mnt/data/content-save/2017-02-01-021116',
        queue_size=1024,
        batch_size=16,
        net_dim=256,
        shorter_dim=List([256, 512]),
        gpu_frac=0.3,
        timeout_fn=Factory.TimeoutFunction.QUARDRATIC(offset=0.02, delta=0.01),
    ),
)
'''

app.add_route(
    url='/classify/food_types',
    factory=Factory(
        working_dir='/mnt/data/food-save/2016-11-25-112622',
        queue_size=1024,
        batch_size=16,
        net_dim=256,
        shorter_dim=List([256, 512]),
        gpu_frac=0.3,
        timeout_fn=Factory.TimeoutFunction.QUARDRATIC(offset=0.02, delta=0.01),
    ),
)
