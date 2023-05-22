import numpy as np

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
FeatureMap1,label1,FeatureMap2,label2,FeatureName,date,ValidFeature = GetFeature_Tra(tra,cfg,Fusion=True)
Feature_tra = np.r_[FeatureMap1, FeatureMap2]
label_tra = np.r_[label1,label2]
Classes = len(ID_tra)
Inputsize = len(FeatureName)
###### 5.Train a classifier
from Core.NeuralMain import CNNMix
model = CNNMix(FeatureMap1,label1,cfg,Inputsize,Classes)
###### 6. Get the weight of each method
from Core.FusionModel import GetValAccs
Accs = GetValAccs(FeatureMap1,label1,ValidFeature,model,Classes,date)
#  Take the average to get the weight of each method
weights_naive = np.mean(Accs[0, :, :], axis=0)
weights_graph = np.mean(Accs[1, :, :], axis=0)
###### 7. Train a Classifier
model = CNNMix(Feature_tra,label_tra,cfg,Inputsize,Classes)
###### 8. Match score calculation
from Core.TrainSetAndTestSet import GetFeature_Tes
Feature_test,label_test = GetFeature_Tes(tes,FeatureName,cfg)
from Core.Fusion import Score,FusionScore
sim = Score(FeatureMap1+FeatureMap2,Feature_test,label_test,K=4,model=model,Classes=Classes)
nm_score = FusionScore(sim,weights_naive,Classes)
gm_score = FusionScore(sim,weights_graph,Classes)
###### 9.User Matching
pre_ind_nai = nm_score.argmax(axis=1)
print('Acuraccy-NM:',sum(label_test == ID_tra[pre_ind_nai]) / len(label_test))
NM = np.vstack((label_test,ID_tra[pre_ind_nai]))
np.savetxt(cfg.savepath+'MatchingResult_Fusion_NM.txt',NM.transpose(1,0),fmt='%d')
from scipy.optimize import linear_sum_assignment
pre_ind_gra = linear_sum_assignment(gm_score, maximize=True)
print('Accuracy-GM:',sum(label_test[pre_ind_gra[0]] == ID_tra[pre_ind_gra[1]]) / len(label_test))
GM = np.vstack((label_test[pre_ind_gra[0]],ID_tra[pre_ind_gra[1]]))
np.savetxt(cfg.savepath+'MatchingResult_NN_GM.txt',GM.transpose(1,0),fmt='%d')
