# *- encoding: utf-8 -*-
"""
    Standard dataset fetching functions,
    and some fast mask utils
    @author: mehdi.rahim@cea.fr
"""

import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.datasets.base import Bunch

def array_to_niis(data, mask):
    """ Converts masked nii 4D array to 4D niimg
    """
    mask_img = nib.load(mask)
    data_ = np.zeros(data.shape[:1] + mask_img.shape)
    data_[:, mask_img.get_data().astype(np.bool)] = data
    data_ = np.transpose(data_, axes=(1, 2, 3, 0))
    return nib.Nifti1Image(data_, mask_img.get_affine())

def array_to_nii(data, mask):
    """ Converts masked nii 3D array to 3D niimg
    """
    mask_img = nib.load(mask)
    data_ = np.zeros(mask_img.shape)
    data_[mask_img.get_data().astype(np.bool)] = data
    return nib.Nifti1Image(data_, mask_img.get_affine())

def set_base_dir():
    """ base_dir
    """
    base_dir = ''
    with open(os.path.join(os.path.dirname(__file__), 'paths.pref'),
              'rU') as f:
        paths = [x.strip() for x in f.read().split('\n')]
        for path in paths:
            if os.path.isdir(path):
                base_dir = path
                break
    if base_dir == '':
        raise OSError('Data not found !')
    return base_dir


def set_data_base_dir(folder):
    """ base_dir + folder
    """
    return os.path.join(set_base_dir(), folder)


def set_cache_base_dir():
    """ memory cache folder
    """
    return set_data_base_dir('tmp')

def set_features_base_dir():
    """ features folder
    """
    return set_data_base_dir('features')

def set_rs_fmri_base_dir():
    """ baseline rs-fmri folder
    """
    return set_data_base_dir('ADNI_baseline_rs_fmri_mri')

def set_fdg_pet_base_dir():
    """ baseline fdg-pet folder
    """
    return set_data_base_dir('ADNI_baseline_fdg_pet')

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
    mmscores = []
    subjects = []
    for f in subject_paths:
        _, subject_id = os.path.split(f)
        if not subject_id in excluded_subjects:
            func_files.append(glob.glob(os.path.join(f, 'func', 'swr*.nii'))[0])
            dx_group.append( \
            s_description[s_description.Subject_ID == subject_id[1:]]\
            .DX_Group_x.values[0])
            subjects.append(subject_id[1:])
            mmscores.append( \
            s_description[s_description.Subject_ID == subject_id[1:]]\
            .MMSCORE.values[0])
    return Bunch(func=func_files, dx_group=dx_group,
                 mmscores=mmscores, subjects=subjects)


def fetch_adni_longitudinal_rs_fmri_DARTEL():
    return fetch_adni_longitudinal_rs_fmri('ADNI_longitudinal_rs_fmri_DARTEL')

def fetch_adni_longitudinal_rs_fmri(dirname='ADNI_longitudinal_rs_fmri'):
    """ Returns paths of ADNI rs-fMRI
    """
    BASE_DIR = set_data_base_dir(dirname)
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, 'I[0-9]*')))
    excluded_images = np.loadtxt(os.path.join(BASE_DIR,
                                                'excluded_subjects.txt'),
                                   dtype=str)
    s_description = pd.read_csv(os.path.join(BASE_DIR,
                                             'description_file.csv'))
    func_files = []
    dx_group = []
    subjects = []
    images = []
    for f in subject_paths:
        _, image_id = os.path.split(f)
        if not image_id in excluded_images:
            func_files.append(glob.glob(os.path.join(f, 'func', 'wr*.nii'))[0])
            dx_group.append( \
            s_description[s_description['Image_ID'] == image_id]\
            ['DX_Group'].values[0])
            images.append(image_id)
            subjects.append(\
            s_description[s_description['Image_ID'] == image_id]\
            ['Subject_ID'].values[0])
    return Bunch(func=func_files, dx_group=dx_group,
                 subjects=subjects, images=images)
    

def fetch_adni_baseline_rs_fmri():
    """ Returns paths of ADNI rs-fMRI 
    """
    BASE_DIR = set_data_base_dir('ADNI_baseline_rs_fmri')
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, '[0-9]*')))
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
            func_files.append(glob.glob(os.path.join(f, 'func', 'wr*.nii'))[0])
            dx_group.append( \
            s_description[s_description['Subject ID'] == subject_id]\
            ['DX Group'].values[0])
            subjects.append(subject_id[1:])
    return Bunch(func=func_files, dx_group=dx_group, subjects=subjects)

def fetch_adni_rs_fmri_conn(filename):
    """Returns paths of ADNI rs-fMRI processed connectivity
    for a given npy file with shape : n_subjects x n_voxels x n_rois
    """
    FEAT_DIR = set_features_base_dir()
    conn_file = os.path.join(FEAT_DIR, 'smooth_preproc', filename)
    if not os.path.isfile(conn_file):
        raise OSError('Connectivity data file not found !')
    dataset = fetch_adni_petmr()
    subj_list = dataset['subjects']
    
    return Bunch(fmri_data=conn_file,
                 dx_group=np.array(dataset['dx_group']),
                 mmscores=np.array(dataset['mmscores']),
                 subjects=subj_list)


def fetch_adni_fdg_pet():
    """Returns paths of ADNI baseline FDG-PET
    """
    BASE_DIR = set_fdg_pet_base_dir()
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, 's[0-9]*')))
    excluded_subjects = np.loadtxt(os.path.join(BASE_DIR,
                                                'excluded_subjects.txt'),
                                   dtype=str)
    s_description = pd.read_csv(os.path.join(BASE_DIR,
                                             'description_file.csv'))
    pet_files = []
    dx_group = []
    mmscores = []
    subjects = []
    for f in subject_paths:
        _, subject_id = os.path.split(f)
        if not subject_id in excluded_subjects:
            pet_files.append(glob.glob(os.path.join(f, 'pet', 'w*.nii'))[0])
            dx_group.append( \
            s_description[s_description.Subject_ID == subject_id[1:]]\
            .DX_Group.values[0])
            subjects.append(subject_id[1:])
            mmscores.append( \
            s_description[s_description.Subject_ID == subject_id[1:]]\
            .MMSCORE.values[0])
    return Bunch(pet=pet_files, dx_group=dx_group,
                 mmscores=mmscores, subjects=subjects)

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
    pet_mmscores = np.array(pet_dataset['mmscores'])
    pet_mmscores = pet_mmscores[pet_idx]
    pet_files = np.array(pet_dataset['pet'])[pet_idx]

    return Bunch(pet=pet_files, dx_group=pet_groups,
                 mmscores=pet_mmscores, subjects=remaining_subjects)

def fetch_adni_petmr():
    """Returns paths of the intersection between PET and FMRI datasets
    """
    pet_dataset = fetch_adni_fdg_pet()
    fmri_dataset = fetch_adni_rs_fmri()
    petmr_subjects = np.intersect1d(pet_dataset['subjects'],
                                    fmri_dataset['subjects'],
                                    assume_unique=True)
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
    pet_mmscores = np.array(pet_dataset['mmscores'])
    petmr_mmscores = pet_mmscores[petmr_idx]
    func_files = np.array(fmri_dataset['func'])[mrpet_idx]
    pet_files = np.array(pet_dataset['pet'])[petmr_idx]

    return Bunch(func=func_files, pet=pet_files, dx_group=petmr_groups,
                 mmscores=petmr_mmscores, subjects=petmr_subjects)

def fetch_adni_masks():
    """Returns paths of masks (pet, fmri, both)
    """
    FEAT_DIR = set_features_base_dir()
    return Bunch(mask_pet=os.path.join(FEAT_DIR, 'masks',
                                       'mask_pet.nii.gz'),
                 mask_fmri=os.path.join(FEAT_DIR, 'masks',
                                        'mask_fmri.nii.gz'),
                 mask_petmr=os.path.join(FEAT_DIR, 'masks',
                                         'mask_petmr.nii.gz'))

def set_group_indices(dx_group):
    """Returns indices for each clinical group
    """
    dx_group = np.array(dx_group)
    idx = {}
    for g in ['AD', 'MCI', 'LMCI', 'EMCI', 'Normal']:
        idx[g] = np.where(dx_group == g)[0]
    for g in ['EMCI', 'LMCI']:
        idx['MCI'] = np.hstack((idx['MCI'], idx[g]))
    return idx
