from model import App, Factory

app = App(__name__)

'''
app.add_route(
    url='/classify/food_types',
    factory=Factory(working_dir='/mnt/data/food-save/2016-11-25-112622'),
)
'''

app.add_route(
    url='/classify/6_categories',
    factory=Factory(working_dir='/mnt/data/content-save/2017-02-22-160750'),
)
