from scipy.io import loadmat
import numpy as np
import pandas as pd
from proteus.predic import sbp
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import cPickle as pickle

from proteus.predic.high_confidence import TwoStagesPrediction
from proteus.predic import prediction
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

connectomes = []
for id_subj in demog.index.values:
    dat = loadmat('/home/cdansereau/data/sbp/sz_6sites_preproc/corrmatrix/corrmatrix_07'+id_subj+'.mat')
    connectomes.append(dat['R'])

connectomes = np.hstack(connectomes).T


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
    subj_id = []
    for train, test in skf.split(np.zeros(len(y)), y):

        crm = prediction.ConfoundsRm(confounds[train], connectomes[train])
        dat_ = crm.transform(confounds[train], connectomes[train])

        tsp = TwoStagesPrediction(gamma=1., gamma_auto_adjust=True,
                                  thresh_ratio=0.03,
                                  min_gamma=0.9,
                                  shuffle_test_split=0.2,
                                  recurrent_modes=6,
                                  n_iter=500)

        tsp.fit_recurrent(dat_, dat_, y[train])

        #print 'ACC ', sbp_cv.score_files(path_data, demog.index.values[test], confounds[test, :], y[test])
        dat_ = crm.transform(confounds[test], connectomes[test])
        scores.append(np.hstack((y[test][:, np.newaxis], tsp.predict(dat_, dat_)[0])))
        subj_id.append(demog.index.values[test])

    scores = np.vstack(scores)
    crm = prediction.ConfoundsRm(confounds, connectomes)
    dat_ = crm.transform(confounds, connectomes)
    tsp.fit_recurrent(dat_, dat_, y)
    data_results = {'scores': scores, 'tsp': tsp, 'crm': crm, 'subj_id': subj_id}
    pickle.dump(data_results, open(file_name, "wb"))


for dyn in [False]:
    for nSubtypes in [3]:
        for nSubtypes_s2 in [3]:
            file_name = '/home/cdansereau/data/sbp/sz_6sites_preproc/results_sz_fmri/connectomes_scale07_split20_recurrent_thres03pct'
            file_name += '_st' + str(nSubtypes)
            file_name += '_2st' + str(nSubtypes_s2)
            if dyn:
                file_name += '_dyn'
            file_name += '.pkl'
            print file_name
            run_acc(dyn, nSubtypes, nSubtypes_s2, file_name=file_name)
