import base64
import os
import re
import requests

URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')
SESSION = requests.Session()
TMP_DIR = '/tmp/img'


def from_path(path):
    print('from_path: {}'.format(path))
    with open(path, 'r') as f:
        content = f.read()
    return content


def from_url(url):
    print('from_url: {}'.format(url))
    response = SESSION.get(url)
    return response.content


def to_path(path, content):
    print('to_path: {}'.format(path))
    with open(path, 'w') as f:
        f.write(content)


class Image(object):
    def __init__(self, uri):
        self.uri = uri
        self.file_name = os.path.basename(uri)
        self.class_name = None

        if URL_REGEX.match(uri) is not None:
            file_path = os.path.join(TMP_DIR, self.file_name)

            if os.path.isfile(file_path):
                self.content = from_path(file_path)
            else:
                if not os.path.isdir(TMP_DIR):
                    os.makedirs(TMP_DIR)
                self.content = from_url(uri)
                to_path(file_path, self.content)
        else:
            self.content = from_path(uri)

    def __repr__(self):
        if self.class_name is None:
            return '<Image {}>'.format(self.file_name)
        else:
            return '<Image {}:{}>'.format(self.class_name, self.file_name)

    def b64_content(self):
        return base64.standard_b64encode(self.content)

    def json(self):
        return {
            'photoName': self.file_name,
            'photoContent': self.b64_content(),
        }

    def classname_from_dict(self, dict_):
        self.class_name = max(dict_['classes'].items(), key=lambda item: float(item[1]))[0]
