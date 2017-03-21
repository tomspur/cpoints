#!/usr/bin/python
# Copyright (c) 2017 Thomas Spura

import pandas as pd
import sys

import critical_points as cp

data = cp.read_namd(sys.argv[1])
print(data)

print("Exiting")
sys.exit(0)

print("Symmetrizing")
data.extrapolate(data.temperature, field_mixing=True)

delta = 0.01
res = []
for i in range(-10, 10):
    data.extrapolate(data.temperature + delta*i)  # , data.observable)# + delta*i)
    res.append([data.temperature + delta*i, data.rew_obs,
                data.K2, data.K4, data.fm_s])
res = pd.DataFrame(res, columns=["temperature", "pressure", "K2", "K4", "s"])

print(res)
res.to_csv("tmp.csv", index=False)
