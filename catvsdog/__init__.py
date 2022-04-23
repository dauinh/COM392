import os

PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(PATH, r'data')

assert os.path.isdir(PATH)
assert os.path.isdir(DATA_PATH)