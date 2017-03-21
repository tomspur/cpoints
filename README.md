critical points
===============

Calculate critical points from monte carlo or molecular dynamics simulations
with interactive reweighting in Python.

Features
========

* Reads NAMD output file and dumps the required data as pickle file to disk
  (as basename.pkl). This file will be tried to read first on further reads of
  the same NAMD output file.
* Interactive reweighting to a new temperature and chemical potential:
  ![](https://raw.githubusercontent.com/tomspur/critical_points/master/docs/interactive-small.gif)
