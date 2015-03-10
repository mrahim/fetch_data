# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:26:51 2015

@author: mehdi.rahim@cea.fr
"""

import fetch_data

print fetch_data.set_base_dir()
print fetch_data.set_data_base_dir('features')
pet = fetch_data.fetch_adni_fdg_pet()

print fetch_data.fetch_adni_masks()