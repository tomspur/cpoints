# Copyright (c) 2017 Thomas Spura

import sys

import cpoints as cp

data = cp.Statistics()
# TODO use argparse
data.from_namd(sys.argv[1])
data.to_csv(sys.argv[2])
