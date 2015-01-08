# *- encoding: utf-8 -*-
"""
    Author : Mehdi Rahim
"""

import os, glob
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch



def set_base_dir():
    """ BASE_DIR could be on disk4t or on FREECOM
    """
    
    base_dir = '/disk4t/mehdi/data/ADNI_baseline_rs_fmri_mri/preprocess_output'
    if not os.path.isdir(base_dir):
        base_dir = '/media/FREECOM/Data/ADNI_baseline_rs_fmri_mri'
        if not os.path.isdir(base_dir):
            base_dir = '/media/mr243268/FREECOM/Data/ADNI_baseline_rs_fmri_mri'
	    if not os.path.isdir(base_dir):
		base_dir = ''
	        raise OSError('Data not found !')
    return base_dir

def fetch_adni_rs_fmri():
    """ Returns paths of ADNI resting-state fMRI
    """
    
    BASE_DIR = set_base_dir()
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, 's[0-9]*')))
    excluded_subjects = np.loadtxt(os.path.join(BASE_DIR,
                                                'excluded_subjects.txt'),
                                   dtype=str)
    
    s_description = pd.read_csv(os.path.join(BASE_DIR,
                                             'description_file.csv'))
    func_files = []
    dx_group = []
    for f in subject_paths:
        _, subject_id = os.path.split(f)
        if not subject_id in excluded_subjects:
            func_files.append(glob.glob(os.path.join(f, 'func', 'twr*.nii'))[0])
            dx_group.append( \
            s_description[s_description.Subject_ID == subject_id[1:]].DX_Group_x.values[0])
    return Bunch(func=func_files, dx_group=dx_group)

