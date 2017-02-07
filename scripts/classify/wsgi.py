import os
import sys

activate_this = '/home/harry/slender/venv/bin/activate_this.py'
execfile(activate_this, dict(__file__=activate_this))

os.environ['PYTHON_EGG_CACHE'] = '/tmp'
sys.path.insert(0, os.path.dirname(__file__))

from service import app as application
