from proteus.predic import sbp
import pandas as pd
import scipy.io
from os import listdir
from sklearn.utils import shuffle
from sklearn import preprocessing
from proteus.predic import clustering as cls
from proteus.predic import stability
from nilearn import plotting
import hcp
from proteus.io import util

import glob,os
import nibabel as nib

template_007= nib.load('/home/cdansereau/data/template_cambridge_basc_multiscale_nii_sym/template_cambridge_basc_multiscale_sym_scale007.nii.gz')
template_012= nib.load('/home/cdansereau/data/template_cambridge_basc_multiscale_nii_sym/template_cambridge_basc_multiscale_sym_scale012.nii.gz')
template_020= nib.load('/home/cdansereau/data/template_cambridge_basc_multiscale_nii_sym/template_cambridge_basc_multiscale_sym_scale020.nii.gz')



base_path = '/home/cdansereau/data/sbp/sz_6sites_preproc/fmri_nii/'
list_of_files = glob.glob(base_path+"*")

demog = pd.read_csv('/home/cdansereau/data/schizo/schizo_6_new_studies_20170516_matched.csv', index_col=0)
demog.index = demog.index.str.strip()
demog.columns = demog.columns.str.strip()
#print list_of_files


import cPickle as pickle

import numpy as np
import pandas as pd
from proteus.predic import sbp
from proteus.predic import subtypes
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

y = demog[['sz']].values.ravel()
# confounds = civet_demog.loc[:, ['gender', 'age', 'fd', 'mean_ct_wb']].values
# confounds = civet_demog.loc[:,['gender']].values
confounds = demog[['sex', 'age', 'rFD1']].values
confounds[:, 1:] = preprocessing.scale(confounds[:, 1:])
confounds[:, 0] = preprocessing.binarize(confounds[:, 0].reshape(-1, 1), threshold=1)[:, 0]

# Estimate model performance
path_data = '/home/cdansereau/data/sbp/sz_6sites_preproc/cambridge20_vox_blur/'

def run_acc(dyn, nSubtypes_fmri, nSubtypes_stage2, file_name):
    skf = StratifiedKFold(10)
    scores = []
    for train, test in skf.split(np.zeros(len(y)), y):
        sbp_cv = sbp.SBP(verbose=True, gamma=.8, gamma_auto_adjust=False, thresh_ratio=0.03, min_gamma=0.85, nSubtypes=nSubtypes_fmri,
                         nSubtypes_stage2=nSubtypes_stage2,
                         stage1_model_type='svm',
                         dynamic=dyn, stage1_metric='accuracy',
                         stage2_metric='f1_weighted',
                         shuffle_test_split=0.8,
                         n_iter=100,
                         s2_branches=True)

        sbp_cv.fit_files(path_data, demog.index.values[train], confounds[train, :], y[train],
                         n_seeds=20, skip_st_training=False)

        print 'ACC ', sbp_cv.score_files(path_data, demog.index.values[test], confounds[test, :], y[test])
        scores.append(sbp_cv.res)

    scores = np.vstack(scores)
    sbp_cv.fit_files(path_data, demog.index.values, confounds, y, n_seeds=20, skip_st_training=False)
    data_results = {'scores': scores, 'sbp': sbp_cv}
    pickle.dump(data_results, open(file_name, "wb"))


for dyn in [False]:
    for nSubtypes in [3]:
        for nSubtypes_s2 in [3]:
            file_name = '/home/cdansereau/data/sbp/sz_6sites_preproc/results_sz_fmri/sbp_res_cam20_split80_noauto'
            file_name += '_st' + str(nSubtypes)
            file_name += '_2st' + str(nSubtypes_s2)
            if dyn:
                file_name += '_dyn'
            file_name += '.pkl'
            print file_name
            run_acc(dyn, nSubtypes, nSubtypes_s2, file_name=file_name)
