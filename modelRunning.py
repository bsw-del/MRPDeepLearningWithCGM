import numpy as np
import pandas as pd
import chardet
from matplotlib import pyplot as plt
import os
from random import randrange
import random
import warnings

from pandas.core.common import SettingWithCopyWarning


class ModelRunner:
    ''' Class to do initial reading and formatting of the source data file'''

    def __init__(self, filename='metrics.csv',path='/Users/bsw/Documents/MRPLocal/DATA/'):
        self.path = path
        self.filename = filename
        self.file_source = os.path.join(path, filename)
        