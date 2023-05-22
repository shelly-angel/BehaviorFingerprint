import numpy as np
from torch.utils.data import DataLoader
from Core.TrainAndTest import Test_Graph
from Core.TrainSetAndTestSet import NN_Input

###### 1.Relevant parameter configuration
from Core.utils import config
cfg = config()
###### 2.Import Data
# Get Known set data
from Core.DataProcessing import ImportDataSet
tra,ID_tra = ImportDataSet(file=cfg.data_path+'IPTV_tra.txt',sep=' ')
# Get Unknown set data
tes,_ = ImportDataSet(file=cfg.data_path+'IPTV_tes.txt',sep=' ')
###### 3.Fixed random seed
from Core.setRandomSeed import set_random_seed
set_random_seed(cfg.seed)
print('Fixed random seed')
###### 4.Get training set features
from Core.TrainSetAndTestSet import GetFeature_Tra
FeatureMap1,label1,FeatureMap2,label2,FeatureName = GetFeature_Tra(tra,cfg)
Feature_tra = np.r_[FeatureMap1, FeatureMap2]
label_tra = np.r_[label1, label2]
Classes = len(ID_tra)
Inputsize = len(FeatureName)
###### 5.Train a classifier
from Core.NeuralMain import CNNMix
model = CNNMix(Feature_tra, label_tra,cfg,Inputsize,Classes)
###### 6.Match score calculation
from Core.TrainSetAndTestSet import GetFeature_Tes
Feature_test,label_test = GetFeature_Tes(tes,FeatureName,cfg)
Input_test = DataLoader(NN_Input(Feature_test, label_test), batch_size=cfg.batch_size, shuffle=False)
score = Test_Graph(model,Input_test,Classes)
###### 7.User Matching
from Core.utils import MatchID
pre_ind_nai, pre_ind_gra,_,_= MatchID(score)
NM = np.vstack((label_test,ID_tra[pre_ind_nai]))
GM = np.vstack((label_test[pre_ind_gra[0]],ID_tra[pre_ind_gra[1]]))
print('Acuraccy-NM:',sum(label_test == ID_tra[pre_ind_nai]) / len(label_test))
print('Accuracy-GM:',sum(label_test[pre_ind_gra[0]] == ID_tra[pre_ind_gra[1]]) / len(label_test))
# Save the result: the first column is the real label, and the second column is the predicted label
np.savetxt(cfg.savepath+'MatchingResult_NN_NM.txt',NM.transpose(1,0),fmt='%d')
np.savetxt(cfg.savepath+'MatchingResult_NN_GM.txt',GM.transpose(1,0),fmt='%d')

