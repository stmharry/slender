#!/usr/bin/python
# -*- encoding: utf-8 -*-

import base64
import collections
import os
import re
import requests
import sys

BAR_CHAR = '█'
MAX_BAR_SIZE = 100

if sys.argv[1] == '6':
    API_URL = 'http://classify.2bite.com:8080/classify/6_categories'
elif sys.argv[1] == 'f':
    API_URL = 'http://dev.2bite.com:8080/classify/food_types'

if len(sys.argv) == 3:
    URIS = [sys.argv[2]]
else:
    URIS = sorted([
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49b7/classify/c_a3d6336e9fdd4f8ead5ac74518877f72.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49df/classify/c_0684bfe01d1a4ff3a6e9ad7387484ec4.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49df/classify/c_20d3016fd00a4b2ebe7152b51e820f00.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49df/classify/c_41e4a4ec09404f61ab59dda3af3ac670.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49df/classify/c_b5ad6140f9c14ec89efa24de87e7c42f.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49df/classify/c_bfdd25bbd5064aa29e1bec417cb6ccc1.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49df/classify/c_c85bfb1dea834ca2bd6f0d109b50bb2c.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49c6/classify/c_3031dcb16ed4459db7c3d30f0daaddec.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49d4/classify/c_1da98ed52a9e4334b84f1b8140f56c58.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49c9/classify/c_8a080438195f48b29f928652a59dc050.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b4a0c/classify/c_215a3b63d78e4523a3c9923b399e0bc6.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b4a0c/classify/c_21a1083efc8b4ee7aaa50681a1eeb1f0.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b4a0c/classify/c_8221d3062b2e4a13b658fac0e1fef19c.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b4a0c/classify/c_967c3b71a867416fad296b6c15ef41f6.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49ca/classify/c_1b4d058e55f6465a86d9158eab49ac1b.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49cf/classify/c_11115e7d14684240aff9101ffdb9ad9c.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49cf/classify/c_3cf35c0d5e7d48a0baa8cb5c196474a0.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49d0/classify/c_00d4d03611c844759612855d505a8ed6.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49d0/classify/c_119249c1f8f441f58ab830f6c866caf3.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49d0/classify/c_14dcc7fdc6944584a5405de3d950b424.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49d0/classify/c_2ca7c0ab409046289b9eeb91dbdc4c9e.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49d0/classify/c_5eb9f37991b5491b8d62d1d34ffc07fd.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49d0/classify/c_7bcc458353f54b6b9fbd7b923e63cf61.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b4af3/classify/c_419fbc7edd83487dae106c7157531412.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b4af3/classify/c_b5aad8bbb3e0464b9ab51a9822f54e4e.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b4a2c/classify/c_280aecce88864411b41a3e05b46c1c59.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b4a2c/classify/c_41dff6c118ba46938caec265ad1f60b5.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b4a2c/classify/c_48ee95589fc04ead94da0f088640c9e4.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49e0/classify/c_0b6d6f19b5ab41f8842ffa5c684e7038.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49e0/classify/c_10ce25e245704b27b1c9cb3e1287fdf7.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49e0/classify/c_1b0d2788503741ac81282a861cb2bb44.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49e0/classify/c_36bf1e0276c24b69aceaf954b5e903d0.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49e0/classify/c_4b53d931b3c449ff92073eb76009c258.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49e0/classify/c_64ccb16a02ef412c91cf5bc0f5b8dafd.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49e0/classify/c_7fc86da06b1d48ee80c5ae50cbbf5bfe.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49e0/classify/c_89a30ba48c4d499e950ff972ffb894ed.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_0ad9461efdf0499d8f84767df1914534.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_10105b3dfde9478886116752fc38fbf3.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_1fdf6fd266fd4b1986194d208ad1e748.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_1ff1bf44c366447da105e2556ce32486.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_25696ebf93b545d993aa1f49cfe2d08d.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_3134443e30614f3aa43ff23b076233fa.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_38170908b37049a1ac0f054d91abb431.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_464802fbad1f4f229020339f1897e5a8.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_6ef257a464574d74934421df65fd0ad3.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_7406544a89084d13967ad60cb1ed54c5.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_ac7de9037f7b4e05ac4310566b55001d.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_ae1a482df31b43b2b45c5f0ccbfb55e3.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_b8c1afb161ea465daf717f2f1f92c236.jpg',
        'http://s3-us-west-1.amazonaws.com/pic.2bite.com/event/5642f19c518f6e735e8b49f3/classify/c_c74e65c315434c0198d01ae492a5c2c3.jpg',
    ])


class Image(object):
    URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')
    SESSION = requests.Session()
    TMP_DIR = '/tmp/img'

    @staticmethod
    def from_local(path):
        with open(path, 'r') as f:
            content = f.read()
        return content

    @staticmethod
    def from_online_and_cache(url, path):
        content = Image.SESSION.get(url).content
        with open(path, 'w') as f:
            f.write(content)
        return content

    def __init__(self, uri):
        if Image.URL_REGEX.match(uri) is None:
            self.path = uri
            self.name = os.path.basename(self.path)
            self.content = Image.from_local(self.path)
        else:
            self.url = uri
            self.name = os.path.basename(self.url)
            self.path = os.path.join(Image.TMP_DIR, self.name)

            if os.path.isfile(self.path):
                self.content = Image.from_local(self.path)
            else:
                if not os.path.isdir(Image.TMP_DIR):
                    os.makedirs(Image.TMP_DIR)
                self.content = Image.from_online_and_cache(self.url, self.path)

    def __repr__(self):
        return '<Image {}>'.format(self.name)

    def b64_content(self):
        return base64.standard_b64encode(self.content)

    def json(self):
        return {
            'photoName': self.name,
            'photoContent': self.b64_content(),
        }


images = map(Image, URIS)
json = [image.json() for image in images]
response = requests.post(API_URL, json=json)
items = response.json(object_pairs_hook=collections.OrderedDict)

for (image, item) in zip(images, items):
    print(image.name)
    for (class_name, prediction) in item['classes'].items():
        prediction = float(prediction)
        bar_size = int(MAX_BAR_SIZE * prediction)
        print('{:>16} | {} {:.4f}'.format(
            class_name,
            BAR_CHAR * bar_size,
            prediction,
        ))
    print('')
