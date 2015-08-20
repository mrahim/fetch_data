# -*- coding: utf-8 -*-
"""
    Some utils functions for :
    - masking
    - diagnosis
    - ...
    @author: mehdi.rahim@cea.fr
"""

import os
import glob
import numpy as np
import nibabel as nib
from datetime import date


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


def _glob_subject_img(subject_path, suffix, first_img=False):
    """ Get subject image (pet, func, ...)
        for a given subject and a suffix
    """

    img_files = sorted(glob.glob(os.path.join(subject_path, suffix)))
    if len(img_files) == 0:
        raise IndexError('Image not found...')
    elif first_img:
        return img_files[0]
    else:
        return img_files


def _set_base_dir():
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


def _set_data_base_dir(folder):
    """ base_dir + folder
    """
    return os.path.join(_set_base_dir(), folder)


def _set_cache_base_dir():
    """ memory cache folder
    """
    return _set_data_base_dir('tmp')


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


def _set_group_indices(dx_group):
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
