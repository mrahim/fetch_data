# -*- coding: utf-8 -*-
"""
    Some utils functions for :
    - masking
    - diagnosis
    - custom shuffle splitting
    - ...
    @author: mehdi.rahim@cea.fr
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from datetime import date
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score


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


def _get_subjects_and_description(base_dir,
                                  prefix,
                                  exclusion_file='excluded_subjects.txt',
                                  description_csv='description_file.csv'):
    """  Returns list of subjects and phenotypic dataframe
    """

    # load files and set dirs
    BASE_DIR = _set_data_base_dir(base_dir)
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, prefix)))

    fname = os.path.join(BASE_DIR, exclusion_file)
    if not os.path.isfile(fname):
        raise OSError('%s not found ...' % fname)
    excluded_subjects = []
    if os.stat(fname).st_size > 0:
        excluded_subjects = np.loadtxt(fname, dtype=str)

    fname = os.path.join(BASE_DIR, description_csv)
    if not os.path.isfile(fname):
        raise OSError('%s not found ...' % fname)
    description = pd.read_csv(fname)

    # exclude bad QC subjects
    excluded_paths = np.array(map(lambda x: os.path.join(BASE_DIR, x),
                                  excluded_subjects))
    subject_paths = np.setdiff1d(subject_paths, excluded_paths)

    # get subject_id
    subjects = [os.path.split(s)[-1] for s in subject_paths]

    return subjects, subject_paths, description


def _glob_subject_img(subject_path, suffix, first_img=False):
    """ Get subject image (pet, func, ...)
        for a given subject and a suffix
    """

    img_files = sorted(glob.glob(os.path.join(subject_path, suffix)))
    if len(img_files) == 0:
        raise IndexError('Image not found in %s' % subject_path)
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
                print('Datadir= %s' % path)
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


def _diff_visits(vis_1, vis_2):
    """Returns a numerical difference between two visits
    """
    # First, we convert visits
    v = map(lambda x: 0 if(x in ['bl', 'sc', 'uns1', 'scmri', 'nv', 'f'])
            else int(x[1:]), [vis_1, vis_2])
    # Then, we substract
    return np.absolute(v[0] - v[1])


def _find_closest_exam_code(viscode, exam_codes):
    """Returns the indice and the code of the current viscode
    """
    ind = np.argwhere(exam_codes == viscode)
    if len(ind) > 0:
        ind = ind[0, 0]
    else:
        diff = [_diff_visits(viscode, e) for e in exam_codes]
        ind = np.argmin(diff)
    return viscode, ind


def _get_vcodes(rid, exam_date, dx):
    """ Returns visit codes of an exam_date of a subject
    """

    vcodes = dx[(dx.RID == rid) & (dx.EXAMDATE == exam_date)]['VISCODE'].values
    vcodes2 = dx[(dx.RID == rid) & (dx.EXAMDATE == exam_date)]['VISCODE2'].values

    if not vcodes.any():
        vcodes = [np.nan]
    if not vcodes2.any():
        vcodes2 = [np.nan]

    return [vcodes[0], vcodes2[0]]


def _get_dx(rid, dx, exam=None, viscode=None, return_code=False):
    """Returns all diagnoses for a given
    rid, depending on exam or viscode (mutually exclusive)
    """

    if exam is not None and viscode is not None:
        raise ValueError('Both exam and viscode are set !')

    if exam is not None:
        dates = dx[dx.RID == rid]['EXAMDATE'].values
        exam_dates = [date(int(d[:4]), int(d[5:7]), int(d[8:])) for d in dates]
    elif viscode is not None:
        if viscode[0] == 'v':  # ADNI1
            exam_codes = dx[dx.RID == rid]['VISCODE'].values
        else:  # ADNI GO/2
            exam_codes = dx[dx.RID == rid]['VISCODE2'].values

    # DXCHANGE
    change = dx[dx.RID == rid]['DXCHANGE'].values
    curren = dx[dx.RID == rid]['DXCURREN'].values
    # change, curren have the same length
    dxchange = [int(np.nanmax([change[k], curren[k]]))
                for k in range(len(curren))]

    if exam is not None and len(exam_dates) > 0:
        exam_date, ind = _find_closest_exam_date(exam, exam_dates)
        # TODO : return exam_date or exam_code ?
        if return_code:
            return exam_date
        else:
            return dxchange[ind]
    elif viscode is not None and len(exam_codes) > 0:
        exam_code, ind = _find_closest_exam_code(viscode, exam_codes)
        if return_code:
            return exam_code
        else:
            return dxchange[ind]
    else:
        return -4


def _get_mmse(rid, mmse):
    """Returns mmse for a given rid
    """
    m = mmse[mmse.RID == rid]['MMSCORE'].dropna().values
    if len(m) > 0:
        return np.median(m)
    else:
        return 0.


def _get_dob(rid, demog):
    """Returns date of birth of a given rid
    """
    yy = demog[demog.RID == rid]['PTDOBYY'].dropna().values
    mm = demog[demog.RID == rid]['PTDOBMM'].dropna().values
    if len(yy) > 0 and len(mm) > 0:
        return date(int(yy[0]), int(mm[0]), 01)
    else:
        return date(1900, 01, 01)


def _get_gender(rid, demog):
    """Returns gender if a given rid
    """
    gender = demog[demog.RID == rid]['PTGENDER'].dropna().values
    if len(gender) > 0:
        return int(gender[0])
    else:
        return -1


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


def _set_classification_data(features, dx_group, groups, return_idx=False):
    """Returns X and y for classification according to the chosen groups
    """
    # get group indices
    dx_idx = _set_group_indices(dx_group)
    # stack the desired indices
    idx_ = []
    for group in groups:
        idx_.extend(dx_idx[group])
    # extract corresponding features and classes (binary-only)
    X = features[idx_, ...]
    y = np.hstack(([1]*len(dx_idx[groups[0]]), [-1]*len(dx_idx[groups[1]])))
    if return_idx:
        return X, y, idx_
    else:
        return X, y


def _set_group_data(features, dx_group, group):
    """Returns X for a given group
    """
    # get group indices
    dx_idx = _set_group_indices(dx_group)
    # extract and return the corresponding features
    return features[dx_idx[group], ...]


def StratifiedSubjectShuffleSplit(dataset, groups, n_iter=100, test_size=.3,
                                  random_state=42):
    """ Stratified ShuffleSplit on subjects
    (train and test size may change depending on the number of acquistions)"""

    idx = _set_group_indices(dataset.dx_group)
    groups_idx = np.hstack([idx[group] for group in groups])

    subjects = np.asarray(dataset.subjects)
    subjects = subjects[groups_idx]

    dx = np.asarray(dataset.dx_group)
    dx = dx[groups_idx]

    # extract unique subject ids and dx
    (subjects_unique_values,
     subjects_unique_indices) = np.unique(subjects, return_index=True)

    # extract indices for the needed groups
    dx_unique_values = dx[subjects_unique_indices]
    y = dx_unique_values

    # generate folds stratified on dx
    sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=test_size,
                                 random_state=random_state)
    ssss = []
    for tr, ts in sss:
        # get training subjects
        subjects_tr = subjects_unique_values[tr]

        # get testing subjects
        subjects_ts = subjects_unique_values[ts]

        # get all subject indices
        train = []
        test = []
        for subj in subjects_tr:
            train.extend(np.where(subjects == subj)[0])
        for subj in subjects_ts:
            test.extend(np.where(subjects == subj)[0])

        # append ssss
        ssss.append([train, test])
    return ssss


def SubjectShuffleSplit(dataset, groups, n_iter=100,
                        test_size=.3, random_state=42):
    """ Specific ShuffleSplit (train on all subject images,
    but test only on one image of the remaining subjects)"""

    idx = _set_group_indices(dataset.dx_group)
    groups_idx = np.hstack((idx[groups[0]],
                            idx[groups[1]]))

    subjects = np.asarray(dataset.subjects)
    subjects = subjects[groups_idx]
    subjects_unique = np.unique(subjects)

    n = len(subjects_unique)
    ss = ShuffleSplit(n, n_iter=n_iter,
                      test_size=test_size, random_state=random_state)

    subj_ss = []
    for train, test in ss:
        train_set = np.array([], dtype=int)
        for subj in subjects_unique[train]:
            subj_ind = np.where(subjects == subj)
            train_set = np.concatenate((train_set, subj_ind[0]))
        test_set = []
        for subj in subjects_unique[test]:
            subj_ind = np.where(subjects == subj)
            test_set.append(subj_ind[0][0])
        test_set = np.asarray(test_set)
        subj_ss.append([train_set, test_set])

    return subj_ss


def _train_and_score(clf, X, y, train, test):
    """ Fit a classifier clf and train set
    and return the accuracy score on test set"""

    clf.fit(X[train], y[train])
    return clf.score(X[test], y[test])


def _train_and_score_2(clf, X, y, train, test):
    """ Fit a classifier clf and train set
    and return the score on test set"""

    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    return accuracy_score(y[test], y_pred)


def _train_and_predict(clf, X, y, train, test):
    """ Fit a classifier clf and train set
    and return predictions on test set"""

    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    return accuracy_score(y[test], y_pred)


def _set_subjects_splits(imgs, subjects, dx_group, groups, n_iter=100,
                         test_size=.25, random_state=42):
    """Returns X, y
    """
    X, y, idx = _set_classification_data(imgs, dx_group, groups,
                                         return_idx=True)
    sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=test_size,
                                 random_state=random_state)
    return X, y, idx, sss


def _set_y_from_dx(dx, target='AD'):
    """ dx is a vector of labels
    """
    dx = np.hstack(dx)
    y = - np.ones(dx.shape[0])
    y[dx == 'AD'] = 1
    return y


def _binarize_dx(dx, target=['AD', 'MCI->AD']):
    """ dx is a vector of labels
    """
    dx = np.hstack(dx)
    y = - np.ones(dx.shape[0])
    for t in target:
        y[dx == t] = 1
    return y


def _binarize_dxs(dxs, target=['AD', 'MCI->AD']):
    """ dxs longitudinal vector of labels
    """
    return np.array([_binarize_dx(dx) for dx in dxs])
