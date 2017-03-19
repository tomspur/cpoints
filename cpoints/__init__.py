# Copyright (c) 2017 Thomas Spura
"""
Tools to calculate critical points from monte carlo or molecular dynamics
simulations.
"""
import numpy as np
import os
import pandas as pd
import pickle
import scipy.optimize as opt
import subprocess

PKL_VERSION = 2


def read_mc(fin):
    """ Read Monte Carlo file and return Statistics object.
    """
    data = Statistics("grand_canonical")
    data.from_mc(fin)
    return data


def read_namd(fin):
    """ Read NAMD file and return Statistics object.
    """
    pkl = fin[:fin.rfind(".")] + ".pkl"
    if os.path.exists(pkl):
        print("WARNING: pickle file from previous read found")
        print("WARNING: reading from %s" % pkl)
        with open(pkl, "rb") as pkl:
            data = pickle.load(pkl)
        if hasattr(data, "PKL_VERSION") and data.PKL_VERSION == PKL_VERSION:
            return data
        else:
            print("WARNING: PKL_VERSION does not match"
                  "reading from NAMD again.")
    data = Statistics()
    data.from_namd(fin)
    return data


def namd_search_col(fin, line, column):
    """ Search in NAMD file for @line and return @column-th column.
    """
    ret = subprocess.check_output("grep -I \"%s\" \"%s\" |"
                                  "awk '{print $%d}'" % (line, fin, column),
                                  shell=True)
    nums = np.fromstring(ret, sep="\n")
    assert len(nums) == 1, ret
    return nums[0]


def namd_get_energy_col(fin, column, skip_percent=0.1):
    """ Search in NAMD file for ENERGY lines and return @column-th column.

    Parameters:
    - column: The column that is returned from the ENERGY strings
    - skip_percent: Percentage of the trajectory that is skipped at the
      beginning.
    """
    ret = subprocess.check_output("grep -I 'ENERGY:' \"%s\" |"
                                  "awk '{print $%d}' |"
                                  "grep -v ENERGY" % (fin, column), shell=True)
    nums = np.fromstring(ret, sep="\n")
    return nums[int(skip_percent * len(nums)):]


class Statistics(object):
    """ Class to investigate statistics of MC/MD simulation.
    """

    def __init__(self, ensemble="NPT"):
        self.PKL_VERSION = PKL_VERSION
        self.timestep = -1
        self.temperature = 0
        self.pressure = 0
        self.freq = -1
        self.fm_s = 0.0
        self.data = pd.DataFrame()
        self.ensemble = ensemble

    def __repr__(self):
        return """Ensemble: %s
temperature: %f
pressure: %f
timestep: %f
data.shape: %s
data.columns: %s
head(data): %s
mean(data): %s

Current values at temp %f and obs %f in the phase space:
K2: %f
K4: %f
""" % (self.ensemble, self.temperature, self.pressure,
       self.timestep, self.data.shape,
       self.data.columns.values, self.data.head(), self.data.mean(),
       self.rew_temperature, self.rew_obs,
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
    def observable(self):
        """ Returns the experimental observable.
        """
        if self.ensemble == "NPT":
            return self.pressure
        else:
            return self.mu

    def extrapolate(self, temperature, obs=None, coexistence=True,
                    field_mixing=False):
        """ Extrapolate to new phase space.

        The new point in the phase space is defined by the argumentss:
        - temperature
        - obs for the second observable

        Parameters:
        - coexistence: Also reweight to coexistence with the equal area rule
        - field_mixing: Also estimate "s" parameter of field mixing
        """
        if obs is None:
            # TODO use linear estimate (or another polynomial) as initial guess
            # Use pressure from simulation
            obs = self.observable
        print("INFO: reweighting to:", temperature, obs)
        self.rew_temperature = temperature
        self.rew_obs = obs
        if field_mixing:
            if self.ensemble == "grand_canonical":
                raise NotImplementedError("field mixing not implemented for"
                                          "grand_canonical")
            # Determine "s" parameter from field mixing
            best = opt.fmin_slsqp(self.Ksums, -3.0,
                                  acc=1e-5, epsilon=1e-5)
            self.fm_s = best[0]
            print("Final value for s", self.fm_s)
        if coexistence:
            # Determine best value for obs with the equal area rule
            best = opt.fmin_slsqp(self.area_parameter, obs,
                                  acc=1e-5, epsilon=1e-5)
            self.rew_obs = best[0]
            print("Final value for obs", self.rew_obs)

    def area_parameter(self, observable):
        """ Calculate equal area parameter...

        with a new test value for the reweighting observable.
        """
        self.rew_obs = observable
        w = self.reweighting
        obs = self.critical_observable
        mean = np.average(obs, weights=w)
        where = np.where(obs < mean)
        left = np.sum(w[where])
        right = np.sum(w) - left
        # print("GOT area", observable, left, right)
        try:
            return abs(left - right)/(left+right)
        except ZeroDivisionError:
            return 1e10

    @property
    def reweighting(self):
        """ Calculate weights for new phase point.
        """
        delta_b = 1/self.rew_temperature - 1/self.temperature
        if self.ensemble == "NPT":
            delta_bp = self.rew_obs/self.rew_temperature - \
                       self.pressure/self.temperature

            ret = np.exp(-delta_b*self.data["total_energy"] +
                         -delta_bp*self.data["volume"])
        else:
            delta_mub = self.rew_obs/self.rew_temperature - \
                        self.mu/self.temperature
            ret = np.exp(-(delta_b)*self.data["total_energy"] + \
                          (delta_mub)*self.data["N"])
        if np.sum(ret) == 0.0:
            return np.ones_like(self.data["total_energy"])
        return ret.values

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

    def Ksums(self, fm_s=None):
        """ Returns sum of both cumulants with new parameter for s.
        """
        # print("GOT s", fm_s)
        if fm_s is not None:
            self.fm_s = fm_s
        return self.K2 + self.K4

    def from_namd(self, fin, skip_percent=0.1):
        """ Read statistical data from NAMD output file.

        Note:
        - After reading from NAMD output file a .pkl file will be written to
          speed up later reading of the same output file.

        Parameters:
        - fin: NAMD output file that is read.
        - skip_percent: Percentage of the trajectory that is skipped at the
          beginning.
        """
        assert self.ensemble == "NPT", "Reading from NAMD, " \
                                       "the ensemble must be NPT"
        self.temperature = namd_search_col(fin, "INITIAL TEMPERATURE", 4)
        self.pressure = namd_search_col(fin, "TARGET PRESSURE", 5)

        # Set current phase point
        self.rew_temperature = self.temperature
        self.rew_obs = self.pressure

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

        pkl = fin[:fin.rfind(".")] + ".pkl"
        self.to_pkl(pkl)

    def from_mc(self, fin, usecols=None, temperature=0.999, mu=-2.831):
        """ Read statistical data from monte carlo output file.

        Parameters:
        - temperature and mu: default to the values of the bundled test file!!
        """
        assert self.ensemble == "grand_canonical", "Reading from monte carlo "\
                                "output, the ensemble must be grand_canonical!"
        if usecols is None:
            usecols = [1, 3]
        self.data["N"] = np.loadtxt(fin, usecols=np.array([usecols[0]]))
        self.data["total_energy"] = np.loadtxt(fin,
                                               usecols=np.array([usecols[1]]))

        self.temperature = temperature
        self.mu = mu
        self.rew_temperature = self.temperature
        self.rew_obs = self.mu

    def from_csv(self, fin):
        """ Load statistical data from CSV file.
        """
        self.data = pd.read_csv(fin)

    def to_csv(self, fout):
        """ Save statistical data to CSV file.
        """
        self.data.to_csv(fout, index=False)

    def to_pkl(self, fout):
        """ Save all data and states to pickle file.
        """
        with open(fout, "wb") as fout:
            pickle.dump(self, fout)

    def interact(self):
        """ Start interactive reweighting in ipython.
        """
        from ipywidgets import interact, FloatSlider

        if self.ensemble == "grand_canonical":
            def f(temperature, mu):
                self.rew_temperature = temperature
                self.rew_obs = mu
                w = self.reweighting
                return self.critical_observable.hist(bins=100,
                                                     weights=w)
            temp = self.temperature
            mu = self.observable
            interact(f, temperature=FloatSlider(min=0.95*temp, max=1.05*temp,
                                                step=0.001, value=temp,
                                                continuous_update=False),
                     mu=FloatSlider(min=1.05*mu, max=0.95*mu,
                                    step=0.001, value=mu,
                                    continuous_update=False))
        else:
            def f(temperature, pressure):
                self.rew_temperature = temperature
                self.rew_obs = pressure
                w = self.reweighting
                return self.critical_observable.hist(bins=100,
                                                     weights=w)
            temp = self.temperature
            press = self.observable
            interact(f, temperature=FloatSlider(min=temp-10.0, max=temp+10.0,
                                                step=0.05, value=temp,
                                                continuous_update=False),
                     pressure=FloatSlider(min=press-10.0, max=press+10.0,
                                          step=0.05, value=press,
                                          continuous_update=False))
