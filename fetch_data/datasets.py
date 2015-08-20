# *- encoding: utf-8 -*-
"""
    Standard dataset fetching functions,
    and some fast mask utils
    @author: mehdi.rahim@cea.fr
"""

import os
import glob
import functools
import numpy as np
import pandas as pd
from datetime import date
from sklearn.datasets.base import Bunch
from _utils.utils import (_set_data_base_dir, _rid_to_ptid, _get_dx,
                          _set_cache_base_dir, _glob_subject_img)


def fetch_adni_rs_fmri():
    """ Returns paths of ADNI resting-state fMRI
    """

    # load files and set dirs
    BASE_DIR = _set_data_base_dir('ADNI_baseline_rs_fmri_mri')
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, 's[0-9]*')))

    fname = os.path.join(BASE_DIR, 'excluded_subjects.txt')
    if not os.path.isfile(fname):
        raise OSError('File not found ...')
    excluded_subjects = []
    if os.stat(fname).st_size > 0:
        excluded_subjects = np.loadtxt(fname, dtype=str)

    fname = os.path.join(BASE_DIR, 'description_file.csv')
    if not os.path.isfile(fname):
        raise OSError('File not found ...')
    description = pd.read_csv(fname)

    # exclude bad QC subjects
    for e in excluded_subjects:
        subject_paths.remove(os.path.join(BASE_DIR, e))

    # get subject_id
    subjects = [os.path.split(s)[-1][1:] for s in subject_paths]

    # get func files
    func_files = map(lambda x: _glob_subject_img(x, suffix='func/swr*.nii',
                                                 first_img=True),
                     subject_paths)

    # get phenotype from csv
    df = description[description['Subject_ID'].isin(subjects)]
    dx_group = np.array(df['DX_Group_x'])
    mmscores = np.array(df['MMSCORE'])

    return Bunch(func=func_files, dx_group=dx_group,
                 mmscores=mmscores, subjects=subjects)


def fetch_adni_longitudinal_hippocampus_volume():
    """ Returns longitudinal hippocampus measures
    """

    BASE_DIR = _set_data_base_dir('ADNI_csv')

    dx_list = np.array(['None',
                        'Normal',
                        'MCI',
                        'AD',
                        'Normal->MCI',
                        'MCI->AD',
                        'Normal->AD',
                        'MCI->Normal',
                        'AD->MCI',
                        'AD->Normal'])

    roster = pd.read_csv(os.path.join(BASE_DIR, 'ROSTER.csv'))
    dx = pd.read_csv(os.path.join(BASE_DIR, 'DXSUM_PDXCONV_ADNIALL.csv'))
    fs = pd.read_csv(os.path.join(BASE_DIR, 'UCSFFSX51_05_20_15.csv'))

    # Extract hippocampus values
    column_idx = np.arange(131, 147)
    cols = ['ST' + str(c) + 'HS' for c in column_idx]
    hipp = fs[cols].values
    idx_num = np.array([~np.isnan(h).all() for h in hipp])
    hipp = hipp[idx_num, :]
    rids = fs['RID'].values[idx_num]
    ptids = [_rid_to_ptid(rid, roster) for rid in rids]
    exams = fs['EXAMDATE'].values[idx_num]
    exams = map(lambda e: date(int(e[:4]), int(e[5:7]), int(e[8:])), exams)

    # Extract diagnosis
    dx_ind = np.array(map(_get_dx, rids, [dx]*len(rids), exams))
    dx_group = dx_list[dx_ind]

    return Bunch(dx_group=dx_group, subjects=ptids, hipp=hipp, exam_date=exams)


def fetch_adni_longitudinal_rs_fmri_DARTEL():
    """ Returns longitudinal func processed with DARTEL
    """
    return fetch_adni_longitudinal_rs_fmri('ADNI_longitudinal_rs_fmri_DARTEL',
                                           'resampled*.nii')


def fetch_adni_longitudinal_rs_fmri(dirname='ADNI_longitudinal_rs_fmri',
                                    prefix='wr*.nii'):
    """ Returns paths of ADNI rs-fMRI
    """

    # load files and set dirs
    BASE_DIR = _set_data_base_dir(dirname)
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, 'I[0-9]*')))

    fname = os.path.join(BASE_DIR, 'excluded_subjects.txt')
    if not os.path.isfile(fname):
        raise OSError('File not found ...')
    excluded_images = []
    if os.stat(fname).st_size > 0:
        excluded_images = np.loadtxt(fname, dtype=str)

    fname = os.path.join(BASE_DIR, 'description_file.csv')
    if not os.path.isfile(fname):
        raise OSError('File not found ...')
    description = pd.read_csv(fname)

    # exclude bad QC subjects
    for e in excluded_images:
        subject_paths.remove(os.path.join(BASE_DIR, e))

    # get image_id
    images = [os.path.split(s)[-1] for s in subject_paths]

    # get func files
    func_files = map(lambda x: _glob_subject_img(x, suffix='func/' + prefix,
                                                 first_img=True),
                     subject_paths)

    # get motion files
    motions = map(lambda x: _glob_subject_img(x, suffix='func/' + 'rp_*.txt',
                                              first_img=True),
                  subject_paths)

    # get phenotype from csv
    df = description[description['Image_ID'].isin(images)]
    dx_group = np.array(df['DX_Group'])
    subjects = np.array(df['Subject_ID'])

    return Bunch(func=func_files, dx_group=dx_group,
                 subjects=subjects, images=images, motions=motions)


def fetch_adni_baseline_rs_fmri():
    """ Returns paths of ADNI rs-fMRI
    """
    BASE_DIR = _set_data_base_dir('ADNI_baseline_rs_fmri')
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
        if subject_id not in excluded_subjects:
            func_files.append(glob.glob(os.path.join(f, 'func', 'wr*.nii'))[0])
            dx_group.append(
                s_description[s_description['Subject ID'] == subject_id]
                ['DX Group'].values[0])
            subjects.append(subject_id[1:])

    return Bunch(func=func_files, dx_group=dx_group, subjects=subjects)


def fetch_adni_rs_fmri_conn(filename):
    """Returns paths of ADNI rs-fMRI processed connectivity
    for a given npy file with shape : n_subjects x n_voxels x n_rois
    """
    FEAT_DIR = _set_data_base_dir('features')
    conn_file = os.path.join(FEAT_DIR, 'smooth_preproc', filename)
    if not os.path.isfile(conn_file):
        raise OSError('Connectivity file not found ...')
    dataset = fetch_adni_petmr()
    subj_list = dataset['subjects']

    return Bunch(fmri_data=conn_file,
                 dx_group=np.array(dataset['dx_group']),
                 mmscores=np.array(dataset['mmscores']),
                 subjects=subj_list)


def fetch_adni_longitudinal_fdg_pet():
    """Returns paths of longitudinal ADNI FDG-PET
    """

    BASE_DIR = _set_data_base_dir('ADNI_longitudinal_fdg_pet')
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, '[0-9]*')))
    subjects = [os.path.split(subject_path)[-1]
                for subject_path in subject_paths]
    description = pd.read_csv(os.path.join(BASE_DIR,
                                           'description_file.csv'))
    pet_files = [sorted(glob.glob(os.path.join(
                 subject_path, 'pet', 'wr*.nii')))
                 for subject_path in subject_paths]
    idx = [0]
    pet_files_all = []
    for pet_file in pet_files:
        idx.append(idx[-1] + len(pet_file))
        pet_files_all.extend(pet_file)

    images = [os.path.split(pet_file)[-1].split('_')[-1][:-4]
              for pet_file in pet_files_all]

    df = description[description['Image_ID'].isin(images)]
    dx_group_all = np.array(df['DX_Group'])
    dx_conv_all = np.array(df['DX_Conv'])
    subjects_all = np.array(df['Subject_ID'])
    ages = np.array(df['Age'])

    imgs = np.array(pet_files_all)
    imgs_baseline = np.array([imgs[i] for i in idx[:-1]])
    dxconv_baseline = [dx_conv_all[i] for i in idx[:-1]]
    dx_baseline = set_group_indices(dxconv_baseline)

    return Bunch(pet=pet_files, pet_all=pet_files_all,
                 pet_baseline=imgs_baseline,
                 dx_group=dx_group_all, dx_conv=dx_conv_all,
                 dx_list_baseline=dxconv_baseline,
                 dx_group_baseline=dx_baseline,
                 subjects=subjects, subjects_idx=np.array(idx),
                 images=images, ages=ages, subjects_all=subjects_all)


def fetch_adni_fdg_pet():
    """Returns paths of ADNI baseline FDG-PET
    """
    BASE_DIR = _set_data_base_dir('ADNI_baseline_fdg_pet')
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
        if subject_id not in excluded_subjects:
            pet_files.append(glob.glob(os.path.join(f, 'pet', 'w*.nii'))[0])
            dx_group.append(
                s_description[s_description.Subject_ID == subject_id[1:]]
                .DX_Group.values[0])
            subjects.append(subject_id[1:])
            mmscores.append(
                s_description[s_description.Subject_ID == subject_id[1:]]
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
        pet_idx.append(
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
        petmr_idx.append(
            np.where(
                np.array(pet_dataset['subjects']) == petmr_subject)[0][0])
        mrpet_idx.append(
            np.where(
                np.array(fmri_dataset['subjects']) == petmr_subject)[0][0])

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
    BASE_DIR = _set_data_base_dir('features/masks')

    return Bunch(mask_pet=os.path.join(BASE_DIR, 'mask_pet.nii.gz'),
                 mask_fmri=os.path.join(BASE_DIR, 'mask_fmri.nii.gz'),
                 mask_pet_longitudinal=os.path.join(BASE_DIR,
                                                    'mask_longitudinal_fdg_pet'
                                                    '.nii.gz'),
                 mask_petmr=os.path.join(BASE_DIR, 'mask_petmr.nii.gz'))


def fetch_atlas(atlas_name):
    """Retruns selected atlas path
        atlas_names values are : msdl, harvard_oxford, juelich, mayo ...
    """
    from nilearn.datasets import fetch_msdl_atlas
    CACHE_DIR = _set_cache_base_dir()
    if atlas_name == 'msdl':
        atlas = fetch_msdl_atlas()['maps']
    elif atlas_name == 'harvard_oxford':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'HarvardOxford-cortl-maxprob-thr0-2mm.nii.gz')
    elif atlas_name == 'juelich':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'Juelich-maxprob-thr0-2mm.nii.gz')
    elif atlas_name == 'mayo':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_68_rois.nii.gz')
    elif atlas_name == 'canica':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_canica_61_rois.nii.gz')
    elif atlas_name == 'canica141':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_canica_141_rois.nii.gz')
    elif atlas_name == 'tvmsdl':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_tv_msdl.nii.gz')
    elif atlas_name == 'kmeans':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_kmeans.nii.gz')
    else:
        raise OSError('Atlas not found !')

    return atlas
