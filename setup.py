# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:06:31 2015

@author: mehdi.rahim@cea.fr
"""

from setuptools import setup
import fetch_data

setup(name='fetch_data',
    version=fetch_data.__version__,
    description='Dataset loader',
    long_description=open('README.md').read(),
    author='Mehdi Rahim',
    author_email='rahim.mehdi@gmail.com',
    packages=['fetch_data'],
    requires = ['numpy', 'pandas'],
)
