# Copyright (c) 2017 Thomas Spura

import os


def get_test_file():
    """ Returns the path to the test data file.
    """
    return os.path.sep.join(__file__.split(os.path.sep)[:-1] +
                            ["GK_histo.dat"])
