# *- encoding: utf-8 -*-
"""
    Standard dataset fetching functions,
    and some fast mask utils
    @author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import pandas as pd
from datetime import date
from joblib import Memory
from sklearn.datasets.base import Bunch
from _utils.utils import (_set_data_base_dir, _rid_to_ptid, _get_dx,
                          _set_cache_base_dir, _glob_subject_img, _ptid_to_rid,
                          _set_group_indices, _get_subjects_and_description,
                          _get_vcodes, _get_dob, _get_gender)


DX_LIST = np.array(['None',
                    'Normal',
                    'MCI',
                    'AD',
                    'Normal->MCI',
                    'MCI->AD',
                    'Normal->AD',
                    'MCI->Normal',
                    'AD->MCI',
                    'AD->Normal'])


def fetch_adni_longitudinal_mmse_score():
    """ Returns longitudinal mmse scores
    """
    BASE_DIR = _set_data_base_dir('ADNI_csv')
    roster = pd.read_csv(os.path.join(BASE_DIR, 'ROSTER.csv'))
    dx = pd.read_csv(os.path.join(BASE_DIR, 'DXSUM_PDXCONV_ADNIALL.csv'))
    fs = pd.read_csv(os.path.join(BASE_DIR, 'MMSE.csv'))

    # extract nans free mmse
    mmse = fs['MMSCORE'].values
    idx_num = fs['MMSCORE'].notnull().values
    mmse = mmse[idx_num]

    # extract roster id
    rids = fs['RID'].values[idx_num]

    # caching dataframe extraction functions
    CACHE_DIR = _set_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'fetch_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _getptidsmmse():
        return [_rid_to_ptid(rid, roster) for rid in rids]

    # get subject id
    ptids = memory.cache(_getptidsmmse)()
    # extract visit code (don't use EXAMDATE ; null for GO/2)
    vcodes = fs['VISCODE'].values
    vcodes = vcodes[idx_num]
    vcodes2 = fs['VISCODE2'].values
    vcodes2 = vcodes2[idx_num]

    def _getdxmmse():
        return map(lambda x, y: DX_LIST[_get_dx(x, dx, viscode=y)],
                   rids, vcodes2)

    # get diagnosis
    dx_group = memory.cache(_getdxmmse)()

    return Bunch(dx_group=dx_group, subjects=np.array(ptids),
                 mmse=mmse, exam_codes=vcodes, exam_codes2=vcodes2)


def fetch_adni_longitudinal_csf_biomarker():
    """ Returns longitudinal csf measures
    """
    BASE_DIR = _set_data_base_dir('ADNI_csv')
    roster = pd.read_csv(os.path.join(BASE_DIR, 'ROSTER.csv'))
    dx = pd.read_csv(os.path.join(BASE_DIR, 'DXSUM_PDXCONV_ADNIALL.csv'))
    csf_files = ['UPENNBIOMK.csv', 'UPENNBIOMK2.csv', 'UPENNBIOMK3.csv',
                 'UPENNBIOMK4_09_06_12.csv', 'UPENNBIOMK5_10_31_13.csv',
                 'UPENNBIOMK6_07_02_13.csv', 'UPENNBIOMK7.csv',
                 'UPENNBIOMK8.csv']
    cols = ['RID', 'VISCODE', 'ABETA', 'PTAU', 'TAU']
    # 3,4,5,7,8
    csf = pd.DataFrame()
    for csf_file in csf_files[2:]:
        fs = pd.read_csv(os.path.join(BASE_DIR, csf_file))
        csf = csf.append(fs[cols])

    # remove nans from csf values
    biom = csf[cols[2:]].values
    idx = np.array([~np.isnan(v).any() for v in biom])
    biom = biom[idx]
    # get phenotype
    vcodes = csf['VISCODE'].values[idx]
    rids = csf['RID'].values[idx]

    # caching dataframe extraction functions
    CACHE_DIR = _set_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'fetch_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _getptidscsf():
        return map(lambda x: _rid_to_ptid(x, roster), rids)
    ptids = memory.cache(_getptidscsf)()

    # get diagnosis
    def _getdxcsf():
        return map(lambda x, y: DX_LIST[_get_dx(x, dx, viscode=y)],
                   rids, vcodes)
    dx_group = memory.cache(_getdxcsf)()

    return Bunch(dx_group=dx_group, subjects=np.array(ptids),
                 csf=biom, exam_codes=vcodes, exam_codes2=vcodes)


def fetch_adni_longitudinal_hippocampus_volume():
    """ Returns longitudinal hippocampus measures
    """

    BASE_DIR = _set_data_base_dir('ADNI_csv')

    roster = pd.read_csv(os.path.join(BASE_DIR, 'ROSTER.csv'))
    dx = pd.read_csv(os.path.join(BASE_DIR, 'DXSUM_PDXCONV_ADNIALL.csv'))
    fs = pd.read_csv(os.path.join(BASE_DIR, 'UCSFFSX51_05_20_15.csv'))

    # extract hippocampus numerical values
    column_idx = np.arange(131, 147)
    cols = ['ST' + str(c) + 'HS' for c in column_idx]
    hipp = fs[cols].values
    idx_num = np.array([~np.isnan(h).all() for h in hipp])
    hipp = hipp[idx_num, :]

    # extract roster id
    rids = fs['RID'].values[idx_num]

    # caching dataframe extraction functions
    CACHE_DIR = _set_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'fetch_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    # get subject id
    def _getptidshippo():
        return [_rid_to_ptid(rid, roster) for rid in rids]
    ptids = memory.cache(_getptidshippo)()

    # extract exam date
    exams = fs['EXAMDATE'].values[idx_num]
    vcodes = fs['VISCODE'].values[idx_num]
    vcodes2 = fs['VISCODE2'].values[idx_num]
    exams = map(lambda e: date(int(e[:4]), int(e[5:7]), int(e[8:])), exams)

    # extract diagnosis
    def _getdxhippo():
        return np.array(map(_get_dx, rids, [dx]*len(rids), exams))
    dx_ind = memory.cache(_getdxhippo)()
    dx_group = DX_LIST[dx_ind]

    return Bunch(dx_group=dx_group, subjects=np.array(ptids),
                 hipp=hipp, exam_dates=exams, exam_codes=vcodes,
                 exam_codes2=vcodes2)


def fetch_adni_longitudinal_rs_fmri_DARTEL():
    """ Returns longitudinal func processed with DARTEL
    """
    return fetch_adni_longitudinal_rs_fmri('ADNI_longitudinal_rs_fmri_DARTEL',
                                           'resampled*.nii')


def fetch_adni_longitudinal_rs_fmri(dirname='ADNI_longitudinal_rs_fmri',
                                    prefix='wr*.nii'):
    """ Returns paths of ADNI rs-fMRI
    """

    # get file paths and description
    images, subject_paths, description = _get_subjects_and_description(
                                         base_dir=dirname, prefix='I[0-9]*')

    # get func files
    func_files = map(lambda x: _glob_subject_img(x, suffix='func/' + prefix,
                                                 first_img=True),
                     subject_paths)

    # get motion files
    motions = None
    # motions = map(lambda x: _glob_subject_img(x, suffix='func/' + 'rp_*.txt',
    # first_img=True), subject_paths)

    # get phenotype from csv
    dx = pd.read_csv(os.path.join(_set_data_base_dir('ADNI_csv'),
                                  'DXSUM_PDXCONV_ADNIALL.csv'))
    roster = pd.read_csv(os.path.join(_set_data_base_dir('ADNI_csv'),
                                      'ROSTER.csv'))
    df = description[description['Image_ID'].isin(images)]
    df = df.sort('Image_ID')
    dx_group = np.array(df['DX_Group'])
    subjects = np.array(df['Subject_ID'])
    exams = np.array(df['EXAM_DATE'])
    exams = map(lambda e: date(int(e[:4]), int(e[5:7]), int(e[8:])), exams)

    # caching dataframe extraction functions
    CACHE_DIR = _set_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'fetch_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _get_ridsfmri():
        return map(lambda s: _ptid_to_rid(s, roster), subjects)
    rids = memory.cache(_get_ridsfmri)()

    def _get_examdatesfmri():
        return map(lambda i: _get_dx(rids[i],
                                     dx, exams[i],
                                     viscode=None,
                                     return_code=True), range(len(rids)))
    exam_dates = memory.cache(_get_examdatesfmri)()

    def _get_viscodesfmri():
        return map(lambda i: _get_vcodes(rids[i], str(exam_dates[i]), dx),
                   range(len(rids)))
    viscodes = np.array(memory.cache(_get_viscodesfmri)())
    vcodes, vcodes2 = viscodes[:, 0], viscodes[:, 1]

    return Bunch(func=func_files, dx_group=dx_group, exam_codes=vcodes,
                 exam_dates=exam_dates, exam_codes2=vcodes2,
                 subjects=subjects, images=images, motions=motions)


def fetch_adni_rs_fmri():
    """ Returns paths of ADNI resting-state fMRI
    """

    # get file paths and description
    subjects, subject_paths, description = _get_subjects_and_description(
                                         base_dir='ADNI_baseline_rs_fmri_mri',
                                         prefix='s[0-9]*')
    # get the correct subject_id
    subjects = [s[1:] for s in subjects]

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


def fetch_adni_longitudinal_fdg_pet():
    """Returns paths of longitudinal ADNI FDG-PET
    """

    # get file paths and description
    (subjects,
     subject_paths,
     description) = _get_subjects_and_description(
                    base_dir='ADNI_longitudinal_fdg_pet',
                    prefix='[0-9]*')

    # get pet files
    pet_files = map(lambda x: _glob_subject_img(x, suffix='pet/wr*.nii',
                                                first_img=False),
                    subject_paths)
    idx = [0]
    pet_files_all = []
    for pet_file in pet_files:
        idx.append(idx[-1] + len(pet_file))
        pet_files_all.extend(pet_file)

    images = [os.path.split(pet_file)[-1].split('_')[-1][:-4]
              for pet_file in pet_files_all]

    # get phenotype from csv
    dx = pd.read_csv(os.path.join(_set_data_base_dir('ADNI_csv'),
                                  'DXSUM_PDXCONV_ADNIALL.csv'))
    roster = pd.read_csv(os.path.join(_set_data_base_dir('ADNI_csv'),
                                      'ROSTER.csv'))
    df = description[description['Image_ID'].isin(images)]
    dx_group_all = np.array(df['DX_Group'])
    dx_conv_all = np.array(df['DX_Conv'])
    subjects_all = np.array(df['Subject_ID'])
    ages = np.array(df['Age'])

    exams = np.array(df['Exam_Date'])
    exams = map(lambda e: date(int(e[:4]), int(e[5:7]), int(e[8:])), exams)

    # caching dataframe extraction functions
    CACHE_DIR = _set_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'fetch_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _get_ridspet():
        return map(lambda s: _ptid_to_rid(s, roster), subjects)
    rids = memory.cache(_get_ridspet)()

    def _get_examdatespet():
        return map(lambda i: _get_dx(rids[i],
                                     dx, exams[i],
                                     viscode=None,
                                     return_code=True), range(len(rids)))
    exam_dates = memory.cache(_get_examdatespet)()

    def _get_viscodespet():
        return map(lambda i: _get_vcodes(rids[i], str(exam_dates[i]), dx),
                   range(len(rids)))
    viscodes = np.array(memory.cache(_get_viscodespet)())
    vcodes, vcodes2 = viscodes[:, 0], viscodes[:, 1]

    return Bunch(pet=pet_files_all,
                 dx_group=dx_group_all, dx_conv=dx_conv_all,
                 images=images, ages=ages, subjects=subjects_all,
                 exam_codes=vcodes, exam_dates=exam_dates, exam_codes2=vcodes2)


def fetch_adni_baseline_rs_fmri():
    """ Returns paths of ADNI rs-fMRI
    """

    # get file paths and description
    subjects, subject_paths, description = _get_subjects_and_description(
                                           base_dir='ADNI_baseline_rs_fmri',
                                           prefix='[0-9]*')

    # get func files
    func_files = map(lambda x: _glob_subject_img(x, suffix='func/wr*.nii',
                                                 first_img=True),
                     subject_paths)

    # get phenotype from csv
    df = description[description['Subject_ID'].isin(subjects)]
    dx_group = np.array(df['DX_Group'])

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


def fetch_adni_fdg_pet():
    """Returns paths of ADNI baseline FDG-PET
    """

    # get file paths and description
    subjects, subject_paths, description = _get_subjects_and_description(
                                          base_dir='ADNI_baseline_fdg_pet',
                                          prefix='s[0-9]*')

    # get the correct subject_id
    subjects = [s[1:] for s in subjects]

    # get pet files
    pet_files = map(lambda x: _glob_subject_img(x, suffix='pet/w*.nii',
                                                first_img=True), subject_paths)
    # get phenotype from csv
    df = description[description['Subject_ID'].isin(subjects)]
    dx_group = np.array(df['DX_Group'])
    mmscores = np.array(df['MMSCORE'])

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

    Returns
    -------
    mask : Bunch containing:
           - mask_pet
           - mask_fmri
           - mask_pet_longitudinal
           - mask_petmr
    """
    BASE_DIR = _set_data_base_dir('features/masks')

    return Bunch(mask_pet=os.path.join(BASE_DIR, 'mask_pet.nii.gz'),
                 mask_fmri=os.path.join(BASE_DIR, 'mask_fmri.nii.gz'),
                 mask_pet_longitudinal=os.path.join(BASE_DIR,
                                                    'mask_longitudinal_fdg_pet'
                                                    '.nii.gz'),
                 mask_petmr=os.path.join(BASE_DIR, 'mask_petmr.nii.gz'),
                 mask_petmr_longitudinal=os.path.join(BASE_DIR,
                                                      'mask_longitudinal_petmr'
                                                      '.nii.gz'),
                 mask_fmri_longitudinal=os.path.join(BASE_DIR,
                                                     'mask_longitudinal_fmri'
                                                     '.nii.gz'))


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


def intersect_datasets(dataset1, dataset2, intersect_on='exam_codes'):
    """Returns the intersection of two dataset Bunches.
        The output is a dataset (Bunch).
        The intersection is on patient id and visit code or date
    """
    if intersect_on not in ['exam_codes', 'exam_dates']:
        raise ValueError('intersect_on should be either '
                         'exam_codes or exam_dates')
        return -1

    if 'subjects' not in dataset1.keys() or 'subjects' not in dataset2.keys():
        raise ValueError('Cannot intersect, Subject ID not found !')
        return -1

    if (intersect_on not in dataset1.keys() or
       intersect_on not in dataset2.keys()):
        raise ValueError('Cannot intersect,' + intersect_on + ' not found !')
        return -1
    return 0


def extract_baseline_dataset(dataset):
    """Returns baseline bunch data of a dataset bunch
    """
    # equivalent keys are : 'sc', 'bl', 'scmri'
    idx = np.hstack((np.where(dataset.exam_codes2 == 'sc'),
                     np.where(dataset.exam_codes2 == 'bl'),
                     np.where(dataset.exam_codes2 == 'scmri'))).ravel()

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
        dataset[k] = dataset[k][idx]

    return dataset


def get_demographics(subjects, exam_dates=None):
    """Returns demographic informations (dob, gender)
    """
    BASE_DIR = _set_data_base_dir('ADNI_csv')
    demog = pd.read_csv(os.path.join(BASE_DIR, 'PTDEMOG.csv'))
    roster = pd.read_csv(os.path.join(BASE_DIR, 'ROSTER.csv'))

    # caching dataframe extraction functions
    CACHE_DIR = _set_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'fetch_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _get_ridsdemo():
        return map(lambda s: _ptid_to_rid(s, roster), subjects)
    rids = np.array(memory.cache(_get_ridsdemo)())

    dobs = map(lambda r: _get_dob(r, demog), rids)
    genders = map(lambda r: _get_gender(r, demog), rids)

    return Bunch(dobs=dobs, genders=genders)
