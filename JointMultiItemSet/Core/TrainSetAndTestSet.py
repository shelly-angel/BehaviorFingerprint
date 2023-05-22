from Core.DataProcessing import RandomChoose,ImportData
import numpy as np
import time
# Get samples overlap
def PrePareData(TrainData,TestData,cfg):
    from Core.Fingerprint import ChooseFeature,UnionAllFea
    start = time.perf_counter()
    FeatureName, m = ChooseFeature(TrainData,Multi=cfg.nGram,ratio=cfg.ratio,max_value_len=cfg.ph)
    Feature_test, label_test = UnionAllFea(TestData,FeatureName,max_value_len=m,Multi=cfg.nGram)
    Time = time.perf_counter()-start
    Feature_train, label_train = UnionAllFea(TrainData,FeatureName,max_value_len=m,Multi=cfg.nGram)
    return Feature_train,label_train,Feature_test,label_test,Time
# Get samples non-overlap
def PrePareData_nonoverlap(TrainData,TestData,cfg):
    from Core.Fingerprint import nonoverlap_ChooseFea,nonoverlap_Fea
    start = time.perf_counter()
    FeatureName, m = nonoverlap_ChooseFea(TrainData,Multi=cfg.nGram,ratio=cfg.ratio,max_value_len=cfg.ph)
    Feature_test, label_test = nonoverlap_Fea(TestData,FeatureName,max_value_len=m,Multi=cfg.nGram)
    Time = time.perf_counter()-start
    Feature_train, label_train = nonoverlap_Fea(TrainData,FeatureName,max_value_len=m,Multi=cfg.nGram)
    return Feature_train,label_train,Feature_test,label_test,Time
# Get Samples-for fusion (1. Random select users; 2. reference set)
def RandomlySelectUser(cfg):
    from Core.setRandomSeed import set_random_seed  # Fixed randomization
    set_random_seed(cfg.seed)
    print('Fixed random seed')
    assert len(cfg.date)==4, print('The date partition format is incorrect')
    userID = RandomChoose(Users=cfg.Num,data_path=cfg.data_path,data_name=cfg.data_name,date=cfg.date)
    return userID
# def GetFeature(cfg,userID):
#     n_gram = cfg.nGram;
#     DataName = cfg.data_name;
#     m = cfg.ph;
#     Train1, Train2, TestData, Classes = ImportData(userID,data_path=cfg.data_path,data_name=DataName)
#     from Core.Fingerprint import ChooseFeature,UnionAllFea
#     FeatureName, max_value_len = ChooseFeature(Train1, Multi=n_gram,ratio=cfg.ratio,max_value_len=m)
#     FeatureName2, _ = ChooseFeature(Train2, Multi=n_gram,ratio=cfg.ratio,max_value_len=m)
#     FeatureName = FeatureName.union(FeatureName2)
#     FeatureMap1, label1 = UnionAllFea(Train1,FeatureName,max_value_len=m,Multi=n_gram)  # training set
#     FeatureMap2, label2 = UnionAllFea(Train2,FeatureName,max_value_len=m,Multi=n_gram)  # validation set
#     from Core.Fingerprint import FeatureVector
#     date, ValidFeature = FeatureVector(Train2,FeatureName,max_value_len=m,Multi=n_gram)
#     X, Y = UnionAllFea(TestData, FeatureName, max_value_len,Multi=n_gram) #test set
#     return FeatureMap1,label1,FeatureMap2,label2,X,Y,FeatureName,Classes,date,ValidFeature

def GetFeature(cfg,userID,Test=False,m=3):
    n_gram = cfg.nGram;
    DataName = cfg.data_name;
    m = cfg.ph;
    Train1, Train2, TestData, Classes = ImportData(userID, data_name=DataName)
    from Core.Fingerprint import ChooseFeature,UnionAllFea
    FeatureName, max_value_len = ChooseFeature(Train1, Multi=n_gram,ratio=cfg.ratio,max_value_len=m)
    FeatureName2, _ = ChooseFeature(Train2, Multi=n_gram,ratio=cfg.ratio,max_value_len=m)
    FeatureName = FeatureName.union(FeatureName2)
    FeatureMap1, label1 = UnionAllFea(Train1,FeatureName,max_value_len=m,Multi=n_gram)
    FeatureMap2, label2 = UnionAllFea(Train2,FeatureName,max_value_len=m,Multi=n_gram)
    if Test:
        X, Y = UnionAllFea(TestData, FeatureName, max_value_len,Multi=n_gram)
        return FeatureMap1,label1,FeatureMap2,label2,Classes,X,Y,FeatureName
    return FeatureMap1,label1,FeatureMap2,label2,Classes

from torch.utils.data import Dataset
import torch
class NN_Input(Dataset):
    def __init__(self,X,Y):
        """
        X: FeatureMap,The input shape is (N,f)
        Y: FeatureMap,The input shape is (N,1)
        """
        self.X = X
        self.Label = Y - 1  # The original userid starts from 1, but the classification comparison in loss, the index starts from 0

    def __len__(self):
        return np.size(self.X, axis=0)

    def __getitem__(self, i):
        return (torch.tensor(self.X[i,:], dtype=torch.float),
                torch.tensor(self.Label[i], dtype=torch.long))
########################################################################
def GetFeature_Tra(Traindata,cfg,Fusion=False):
    import pandas as pd
    pd.options.mode.chained_assignment = None
    from Core.Fingerprint import ChooseFeature, UnionAllFea,FeatureVector
    tra = Traindata[(Traindata['date'] > -1) & (Traindata['date'] <= 10)] # ChainWarning
    val = Traindata[(Traindata['date'] > 10) & (Traindata['date'] <= 20)]
    FeatureName,_ = ChooseFeature(tra, Multi=cfg.nGram, ratio=cfg.ratio, max_value_len=cfg.ph)
    FeatureName2,_ = ChooseFeature(val, Multi=cfg.nGram, ratio=cfg.ratio, max_value_len=cfg.ph)
    FeatureName = FeatureName.union(FeatureName2)
    FeatureMap1, label1 = UnionAllFea(tra,FeatureName,max_value_len=cfg.ph,Multi=cfg.nGram)  # training set
    FeatureMap2, label2 = UnionAllFea(val,FeatureName,max_value_len=cfg.ph,Multi=cfg.nGram)  # validation set
    if Fusion:
        date, ValidFeature = FeatureVector(val, FeatureName, max_value_len=cfg.ph, Multi=cfg.nGram)
        return FeatureMap1,label1,FeatureMap2,label2,FeatureName,date,ValidFeature
    return FeatureMap1,label1,FeatureMap2,label2,FeatureName
def GetFeature_Tes(data,FeatureName,cfg):
    from Core.Fingerprint import UnionAllFea
    X, Y = UnionAllFea(data,FeatureName,cfg.ph,Multi=cfg.nGram) #test set
    return X,Y