#!/usr/bin/python
# Copyright (c) 2017 Thomas Spura

import pandas as pd
import sys

import cpoints as cp

data = cp.read_namd(sys.argv[1])
print(data)

delta = 0.01
res = []
for i in range(-10, 10):
    data.extrapolate(data.temperature + delta*i, data.observable)# + delta*i)
    res.append([data.temperature + delta*i, data.observable + delta*i,
                data.K2, data.K4])
res = pd.DataFrame(res, columns=["temperature", "pressure", "K2", "K4"])

print(res)
