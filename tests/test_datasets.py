# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:26:51 2015

@author: mehdi.rahim@cea.fr
"""

import os
import fetch_data
from nose.tools import assert_true


print os.path.dirname(__file__)


def test_fetch_data():
    """test that paths exist
    """

    assert_true(os.path.isdir(fetch_data.set_base_dir))
    assert_true(os.path.isdir(fetch_data.set_cache_base_dir))
    assert_true(os.path.isdir(fetch_data.set_fdg_pet_base_dir()))
