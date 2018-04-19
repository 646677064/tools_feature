"""Set up paths """

import os
import sys

def add_path(path):
  if path not in sys.path:
    sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

# Add lib to PYTHONPATH
#caffe_path = os.path.abspath(os.path.join(this_dir, '..', '..', '..', 'python'))
caffe_path = "/storage2/liushuai/gs6_env/caffe-reid/python/"#os.path.abspath(os.path.join(this_dir, '..', '..', '..', 'python'))
print 'caffe path : {}'.format(caffe_path)
add_path(caffe_path)
