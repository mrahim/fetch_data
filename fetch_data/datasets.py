# *- encoding: utf-8 -*-
"""
    Standard dataset fetching functions,
    and some fast mask utils
    @author: mehdi.rahim@cea.fr
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from datetime import date
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
        if subject_id not in excluded_subjects:
            func_files.append(
                glob.glob(os.path.join(f, 'func', 'swr*.nii'))[0])
            dx_group.append(
                s_description[s_description.Subject_ID == subject_id[1:]]
                .DX_Group_x.values[0])
            subjects.append(subject_id[1:])
            mmscores.append(
                s_description[s_description.Subject_ID == subject_id[1:]]
                .MMSCORE.values[0])
    return Bunch(func=func_files, dx_group=dx_group,
                 mmscores=mmscores, subjects=subjects)



def _rid_to_ptid(rid, roster):
    """Returns patient id for a given rid
    """

    ptid = roster[roster.RID == rid]['PTID'].values
    if len(ptid) > 0:
        return ptid[0]
    else:
        return ''


def _ptid_to_rid(ptid, roster):
    """Returns roster id for a given patient id
    """

    rid = roster[roster.PTID == ptid]['RID'].values
    if len(rid) > 0:
        return rid[0]
    else:
        return ''


def _find_closest_exam_date(acq_date, exam_dates):
    """Returns closest date and indice of the
    closest exam_date from acq_date"""

    diff = [abs(acq_date - e) for e in exam_dates]
    ind = np.argmin(diff)
    return exam_dates[ind], ind


def _get_dx(rid, dx, exam=None):
    """Returns all diagnoses for a given
    rid and sid"""

    dates = dx[dx.RID == rid]['EXAMDATE'].values
    exam_dates = [date(int(d[:4]), int(d[5:7]), int(d[8:])) for d in dates]

    # DXCHANGE
    change = dx[dx.RID == rid]['DXCHANGE'].values
    curren = dx[dx.RID == rid]['DXCURREN'].values

    # change, curren have the same length
    dxchange = [int(np.nanmax([change[k], curren[k]]))
                for k in range(len(curren))]

    if exam is not None and len(exam_dates) > 0:
        exam_date, ind = _find_closest_exam_date(exam, exam_dates)
        # TODO : return exam_date?
        return dxchange[ind]
    else:
        return -4


def fetch_adni_longitudinal_hippocampus_volume():
    """ Returns longitudinal hippocampus measures
    """

    BASE_DIR = set_data_base_dir('ADNI_csv')

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
    motions = []
    for f in subject_paths:
        _, image_id = os.path.split(f)
        if image_id not in excluded_images:
            func_files.append(glob.glob(os.path.join(f, 'func', prefix))[0])
            motions.append(glob.glob(os.path.join(f, 'func', 'rp_*.txt'))[0])
            dx_group.append(
                s_description[s_description['Image_ID'] == image_id]
                ['DX_Group'].values[0])
            images.append(image_id)
            subjects.append(
                s_description[s_description['Image_ID'] == image_id]
                ['Subject_ID'].values[0])

    return Bunch(func=func_files, dx_group=dx_group,
                 subjects=subjects, images=images, motions=motions)


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


def fetch_adni_longitudinal_fdg_pet():
    """Returns paths of longitudinal ADNI FDG-PET
    """

    BASE_DIR = set_data_base_dir('ADNI_longitudinal_fdg_pet')

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
    FEAT_DIR = set_features_base_dir()

    return Bunch(mask_pet=os.path.join(FEAT_DIR, 'masks',
                                       'mask_pet.nii.gz'),
                 mask_fmri=os.path.join(FEAT_DIR, 'masks',
                                        'mask_fmri.nii.gz'),
                 mask_pet_longitudinal=os.path.join(FEAT_DIR, 'masks',
                                                    'mask_longitudinal_fdg_pet'
                                                    '.nii.gz'),
                 mask_petmr=os.path.join(FEAT_DIR, 'masks',
                                         'mask_petmr.nii.gz'))


def set_group_indices(dx_group):
    """Returns indices for each clinical group
    """
    dx_group = np.array(dx_group)
    idx = {}
    for g in ['AD', 'MCI', 'LMCI', 'EMCI', 'Normal', 'MCI-Converter',
              'Normal->MCI']:
        idx[g] = np.where(dx_group == g)[0]
    for g in ['EMCI', 'LMCI']:
        idx['MCI'] = np.hstack((idx['MCI'], idx[g]))
    idx['AD-rest'] = np.hstack((idx['MCI'], idx['Normal']))
    idx['MCI-rest'] = np.hstack((idx['AD'], idx['Normal']))
    idx['Normal-rest'] = np.hstack((idx['AD'], idx['MCI']))

    return idx


def fetch_atlas(atlas_name):
    """Retruns selected atlas path
        atlas_names values are : msdl, harvard_oxford, juelich, mayo ...
    """
    from nilearn.datasets import fetch_msdl_atlas
    CACHE_DIR = set_cache_base_dir()
    if atlas_name == 'msdl':
        atlas = fetch_msdl_atlas()['maps']
    elif atlas_name == 'harvard_oxford':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'HarvardOxford-cortl-maxprob-thr0-2mm.nii.gz')
    elif atlas_name == 'juelich':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'Juelich-maxprob-thr0-2mm.nii.gz')
    elif atlas_name == 'mayo':
        atlas = os.path.join(CACHE_DIR, 'atlas', 'atlas_68_rois.nii.gz')
    elif atlas_name == 'canica':
        atlas = os.path.join(CACHE_DIR, 'atlas', 'atlas_canica_61_rois.nii.gz')
    elif atlas_name == 'canica141':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_canica_141_rois.nii.gz')
    elif atlas_name == 'tvmsdl':
        atlas = os.path.join(CACHE_DIR, 'atlas', 'atlas_tv_msdl.nii.gz')
    elif atlas_name == 'kmeans':
        atlas = os.path.join(CACHE_DIR, 'atlas', 'atlas_kmeans.nii.gz')

    return atlas
