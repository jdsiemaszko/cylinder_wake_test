import os
import sys
import yaml
import pandas as pd
import numpy as np
import csv
import re

filename = 'blobs_distribution_022500.csv'

case_dir = os.getcwd()
input_dir = os.path.join(case_dir,"input_data")
data_dir = os.path.join(case_dir,"data")
plots_dir = os.path.join(case_dir,"plots")

input_path = os.path.join(input_dir, filename)

arg = sys.argv
configFile = arg[1]
if len(arg) > 2:
    raise Exception("More than two arguments inserted!")

#Config the yaml file
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))
config = yaml.load(open(os.path.join(configFile)),Loader=loader)

