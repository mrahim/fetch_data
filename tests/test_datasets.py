# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:26:51 2015

@author: mehdi.rahim@cea.fr
"""

from nose.tools import assert_equal
import os
import numpy as np
import fetch_data


def test_fetach_data():
    # test that paths exist

    assert os.path.isdir(fetch_data.set_base_dir)
    assert os.path.isdir(fetch_data.set_cache_base_dir)
    assert os.path.isdir(fetch_data.set_fdg_pet_base_dir())
