# *- encoding: utf-8 -*-
"""
    Standard dataset fetching functions
    
    @author: mehdi.rahim@cea.fr
"""

import os, glob
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch


def set_features_base_dir():
    """ BASE_DIR could be on disk4t or on FREECOM
    """
    
    base_dir = '/disk4t/mehdi/data/features'
    if not os.path.isdir(base_dir):
        base_dir = '/home/mr243268/data/features'
        if not os.path.isdir(base_dir):
            base_dir = '/media/FREECOM/Data/features'
            if not os.path.isdir(base_dir):
                base_dir = '/media/mr243268/FREECOM/Data/features'
                if not os.path.isdir(base_dir):
                    base_dir = ''
                    raise OSError('Data not found !')
    return base_dir


def set_rs_fmri_base_dir_old():
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



def set_rs_fmri_base_dir():
    """ BASE_DIR could be on disk4t or on FREECOM
    """
    
    base_dir = '/disk4t/mehdi/data/ADNI_baseline_rs_fmri_mri/preprocessed_rs_fmri'
    if not os.path.isdir(base_dir):
        base_dir = '/media/FREECOM/Data/ADNI_rs_fmri'
        if not os.path.isdir(base_dir):
            base_dir = '/media/mr243268/FREECOM/Data/ADNI_rs_fmri'
	    if not os.path.isdir(base_dir):
		base_dir = ''
	        raise OSError('Data not found !')
    return base_dir



def set_fdg_pet_base_dir():
    """ BASE_DIR could be on disk4t or on FREECOM
    """
    
    base_dir = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'
    if not os.path.isdir(base_dir):
        base_dir = '/media/FREECOM/Data/ADNI_baseline_fdg_pet'
        if not os.path.isdir(base_dir):
            base_dir = '/media/mr243268/FREECOM/Data/ADNI_baseline_fdg_pet'
	    if not os.path.isdir(base_dir):
		base_dir = ''
	        raise OSError('Data not found !')
    return base_dir

def fetch_adni_rs_fmri():
    """ Returns paths of ADNI resting-state fMRI
    """
    
    BASE_DIR = set_rs_fmri_base_dir()
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, 's[0-9]*')))
    excluded_subjects = np.loadtxt(os.path.join(BASE_DIR,
                                                'excluded_subjects.txt'),
                                   dtype=str)
    
    s_description = pd.read_csv(os.path.join(BASE_DIR,
                                             'description_file.csv'))
    func_files = []
    dx_group = []
    subjects = []
    for f in subject_paths:
        _, subject_id = os.path.split(f)
        if not subject_id in excluded_subjects:
            func_files.append(glob.glob(os.path.join(f, 'func', 'swr*.nii'))[0])
            dx_group.append( \
            s_description[s_description.Subject_ID == subject_id[1:]].DX_Group_x.values[0])
            subjects.append(subject_id[1:])
    return Bunch(func=func_files, dx_group=dx_group, subjects=subjects)


def fetch_adni_fdg_pet():
    """ Returns paths of ADNI FDG-PET
    """

    BASE_DIR = set_fdg_pet_base_dir()
    s_description = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))
    
    pet_files = []
    dx_group = []       
    subjects = []
    for idx, row in s_description.iterrows():
        pet_files.append(glob.glob(os.path.join(BASE_DIR,
                                                'I' + str(row.Image_ID),
                                                'wI*.nii'))[0])
        dx_group.append(row['DX_Group'])
        subjects.append(row['Subject_ID'])
        
    return Bunch(pet=pet_files, dx_group=dx_group, subjects=subjects)
    


def fetch_adni_fdg_pet_diff():
    """Returns paths of the diff between PET and fMRI datasets
    """
    pet_dataset = fetch_adni_fdg_pet()
    fmri_dataset = fetch_adni_rs_fmri()
    
    remaining_subjects = np.setdiff1d(pet_dataset['subjects'],
                                      fmri_dataset['subjects'])
    pet_idx = []
    for pet_subject in remaining_subjects:
        pet_idx.append(\
        np.where(np.array(pet_dataset['subjects']) == pet_subject)[0][0])
    
    pet_idx = np.array(pet_idx, dtype=np.intp)
    pet_groups = np.array(pet_dataset['dx_group'])
    pet_groups = pet_groups[pet_idx]
    
    pet_files = np.array(pet_dataset['pet'])[pet_idx]

    return Bunch(pet=pet_files, dx_group=pet_groups,
                 subjects=remaining_subjects)


def fetch_adni_petmr():
    """Returns paths of the intersection between PET and FMRI datasets
    """
    pet_dataset = fetch_adni_fdg_pet()
    fmri_dataset = fetch_adni_rs_fmri()
    
    petmr_subjects = np.intersect1d(pet_dataset['subjects'],
                                    fmri_dataset['subjects'],
                                    assume_unique=True)
    
    #remaining_subjects = np.setdiff1d(fmri_dataset['subjects'], petmr_subjects)
    
    petmr_idx = []
    mrpet_idx = []
    for petmr_subject in petmr_subjects:
        petmr_idx.append(\
        np.where(np.array(pet_dataset['subjects']) == petmr_subject)[0][0])
        mrpet_idx.append(\
        np.where(np.array(fmri_dataset['subjects']) == petmr_subject)[0][0])
    
    petmr_idx = np.array(petmr_idx, dtype=np.intp)
    mrpet_idx = np.array(mrpet_idx, dtype=np.intp)
    pet_groups = np.array(pet_dataset['dx_group'])
    petmr_groups = pet_groups[petmr_idx]
    
    func_files = np.array(fmri_dataset['func'])[mrpet_idx]
    pet_files = np.array(pet_dataset['pet'])[petmr_idx]

    return Bunch(func=func_files, pet=pet_files,
                 dx_group=petmr_groups, subjects=petmr_subjects)

def fetch_adni_masks():
    FEAT_DIR = set_features_base_dir()
    return Bunch(mask_pet=os.path.join(FEAT_DIR, 'masks', 'mask_pet.nii.gz'),
                 mask_fmri=os.path.join(FEAT_DIR, 'masks', 'mask_fmri.nii.gz'),
                 mask_petmr=os.path.join(FEAT_DIR, 'masks', 'mask_petmr.nii.gz'))