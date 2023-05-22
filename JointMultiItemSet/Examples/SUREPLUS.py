import numpy as np
###### 1.Relevant parameter configuration
from Core.utils import config
cfg = config()
###### 2.Import Data
# Get Known set data
from Core.DataProcessing import ImportDataSet
tra,_ = ImportDataSet(file=cfg.data_path+'IPTV_tra.txt',sep=' ')
# Get Unknown set data
tes,_ = ImportDataSet(file=cfg.data_path+'IPTV_tes.txt',sep=' ')

###### 3.GetFeature
from Core.TrainSetAndTestSet import PrePareData
Feature_train,label_train,Feature_test,label_test,t = PrePareData(tra,tes,cfg)

###### 4.Similarity calculation
def sim(Feature_train,Feature_test,method='JS'):
    from scipy.spatial.distance import cdist
    if method == 'Jac':
        # (1) Jaccard
        score = 1 - cdist(np.array(Feature_test, dtype=bool), np.array(Feature_train, dtype=bool), 'jaccard')
    elif method == 'JS':
        # (2)KL
        score = 2 - np.square(cdist(Feature_test, Feature_train, 'jensenshannon')) * 2
    elif method == 'JKL':
        # (3)JKL
        Jaccard = 1 - cdist(np.array(Feature_test, dtype=bool), np.array(Feature_train, dtype=bool), 'jaccard')
        KL = np.square(cdist(Feature_test, Feature_train, 'jensenshannon')) * 2
        score = Jaccard / (KL + 1e-6)
    else:
        score=0
        print('similarity error,please define sim function!')
    return score
score = sim(Feature_train,Feature_test,method='JKL')

###### 5.User matching
from Core.utils import MatchID
pre_ind_nai, pre_ind_gra,_,_= MatchID(score)
NM = np.vstack((label_test,label_train[pre_ind_nai]))
GM = np.vstack((label_test[pre_ind_gra[0]],label_train[pre_ind_gra[1]]))
print('Acuraccy-NM:',sum(label_test == label_train[pre_ind_nai]) / len(label_test))
print('Accuracy-GM:',sum(label_test[pre_ind_gra[0]] == label_train[pre_ind_gra[1]]) / len(label_test))
# Save the result: the first column is the real label, and the second column is the predicted label
np.savetxt(cfg.savepath+'MatchingResult_Sim_NM.txt',NM.transpose(1,0),fmt='%d')
np.savetxt(cfg.savepath+'MatchingResult_Sim_GM.txt',GM.transpose(1,0),fmt='%d')

