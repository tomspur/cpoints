# Copyright (c) 2017 Thomas Spura
"""
Tools to calculate critical points from monte carlo or molecular dynamics
simulations.
"""
import pandas as pd
import numpy as np
import subprocess


def namd_search_col(fin, line, column):
    """ Search in NAMD file for @line and return @column-th column.
    """
    ret = subprocess.check_output("grep \"%s\" \"%s\" |"
                                  "awk '{print $%d}'" % (line, fin, column),
                                  shell=True)
    nums = np.fromstring(ret, sep="\n")
    assert len(nums) == 1
    return nums[0]


def namd_get_energy_col(fin, column, skip_percent=0.1):
    """ Search in NAMD file for ENERGY lines and return @column-th column.

    Parameters:
    - column: The column that is returned from the ENERGY strings
    - skip_percent: Percentage of the trajectory that is skipped at the
      beginning.
    """
    ret = subprocess.check_output("grep 'ENERGY:' \"%s\" |"
                                  "awk '{print $%d}' |"
                                  "grep -v ENERGY" % (fin, column), shell=True)
    nums = np.fromstring(ret, sep="\n")
    return nums[int(skip_percent * len(nums)):]


class Statistics(object):
    """ Class to investigate statistics of MC/MD simulation.
    """

    def __init__(self, ensemble="NPT"):
        self.timestep = -1
        self.freq = -1
        self.fm_s = 0.0
        self.data = pd.DataFrame()
        self.ensemble = ensemble

    def __repr__(self):
        return """Ensemble: %s
timestep: %f
data.shape: %s
data.columns: %s
head(data): %s
mean(data): %s

Current values at this point in the phase space:
K2: %f
K4: %f
""" % (self.ensemble, self.timestep, self.data.shape,
       self.data.columns.values, self.data.head(), self.data.mean(),
       self.K2, self.K4)

    @property
    def critical_observable(self):
        """ Returns the critical observable.
        """
        if self.ensemble == "NPT":
            return self.data["density"] - \
                self.fm_s * self.data["total_energy"] / self.data["volume"]
        else:
            return self.data["N"]

    @property
    def reweighting(self):
        return np.ones_like(self.data["total_energy"])

    @property
    def K2(self):
        """ Returns second cumulant.
        """
        w = self.reweighting
        avg = np.average(self.critical_observable, weights=w)
        H_m = np.abs(self.critical_observable - avg)
        M1 = np.average(H_m, weights=w)
        H_m2 = H_m**2
        M2 = np.average(H_m2, weights=w)
        K2 = M2/(M1*M1)
        return K2

    @property
    def K4(self):
        """ Returns fourth cumulant.
        """
        w = self.reweighting
        avg = np.average(self.critical_observable, weights=w)
        H_m = np.abs(self.critical_observable - avg)
        H_m2 = H_m**2
        M2 = np.average(H_m2, weights=w)
        M4 = np.average(H_m2**2, weights=w)
        K4 = M4/(M2*M2)
        return K4

    def from_namd(self, fin, skip_percent=0.1):
        """ Read statistical data from NAMD output file.

        Parameters:
        - fin: NAMD output file that is read.
        - skip_percent: Percentage of the trajectory that is skipped at the
          beginning.
        """
        assert self.ensemble == "NPT", "Reading from NAMD, " \
                                       "the ensemble must be NPT"
        self.timestep = namd_search_col(fin, "Info: TIMESTEP", 3)
        self.freq = namd_search_col(fin, "PRESSURE OUTPUT STEPS", 5)
        self.mass = namd_search_col(fin, "TOTAL MASS", 5)
        self.data["volume"] = namd_get_energy_col(fin, 19,
                                                  skip_percent=skip_percent)
        self.data["density"] = self.mass/self.data["volume"]
        self.data["pressure"] = namd_get_energy_col(fin, 17,
                                                    skip_percent=skip_percent)
        self.data["temperature"] = \
            namd_get_energy_col(fin, 13,
                                skip_percent=skip_percent)
        self.data["av_pressure"] = namd_get_energy_col(
            fin, 20, skip_percent=skip_percent)
        self.data["av_temperature"] = namd_get_energy_col(
            fin, 16, skip_percent=skip_percent)
        self.data["total_energy"] = namd_get_energy_col(
            fin, 12, skip_percent=skip_percent)
        self.data["elect_energy"] = namd_get_energy_col(
            fin, 7, skip_percent=skip_percent)
        # self.data["vdw_energy"] = namd_get_energy_col(
        #     fin, 8, skip_percent=skip_percent)
        self.data["kinetic_energy"] = namd_get_energy_col(
            fin, 11, skip_percent=skip_percent)

    def from_mc(self, fin, usecols=None):
        """ Read statistical data from monte carlo output file.
        """
        assert self.ensemble == "grand_canonical", "Reading from monte carlo "\
                                "output, the ensemble must be grand_canonical!"
        if usecols is None:
            usecols = [1, 3]
        self.data["N"] = np.loadtxt(fin, usecols=np.array([usecols[0]]))
        self.data["E"] = np.loadtxt(fin, usecols=np.array([usecols[1]]))

    def from_csv(self, fin):
        """ Load statistical data from CSV file.
        """
        self.data = pd.read_csv(fin)

    def to_csv(self, fout):
        """ Save statistical data to CSV file.
        """
        self.data.to_csv(fout, index=False)
