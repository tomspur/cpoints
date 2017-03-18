#!/usr/bin/python
# Copyright (c) 2017 Thomas Spura

import sys

import cpoints as cp

data = cp.read_namd(sys.argv[1])
print(data)
