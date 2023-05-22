import pandas as pd
from torch.utils.data import DataLoader
from Core.DataProcessing import ImportData
from Core.Fusion import SingleMatcherAcc,fusionAcc
from Core.Fusion import AssignWeights
import torch
import numpy as np
import torch.nn.functional as F
from Core.utils import config
cfg = config()
savepath= cfg.savepath
data_dir = cfg.data_path

def GetFeature(cfg,userID):
    n_gram = cfg.nGram;
    DataName = cfg.data_name;
    m = cfg.ph;
    Train1, Train2, TestData, Classes = ImportData(userID,data_path=cfg.data_path,data_name=DataName)
    from Core.Fingerprint import ChooseFeature,UnionAllFea
    FeatureName, max_value_len = ChooseFeature(Train1, Multi=n_gram,ratio=cfg.ratio,max_value_len=m)
    FeatureName2, _ = ChooseFeature(Train2, Multi=n_gram,ratio=cfg.ratio,max_value_len=m)
    FeatureName = FeatureName.union(FeatureName2)
    FeatureMap1, label1 = UnionAllFea(Train1,FeatureName,max_value_len=m,Multi=n_gram)  # training set
    FeatureMap2, label2 = UnionAllFea(Train2,FeatureName,max_value_len=m,Multi=n_gram)  # validation set
    from Core.Fingerprint import FeatureVector
    date, ValidFeature = FeatureVector(Train2,FeatureName,max_value_len=m,Multi=n_gram) # cross validation
    X, Y = UnionAllFea(TestData, FeatureName, max_value_len,Multi=n_gram) # test set
    return FeatureMap1,label1,FeatureMap2,label2,X,Y,FeatureName,Classes,date,ValidFeature
def Test_Graph(model,data,classes):
    model.eval()
    predict = np.zeros((classes,classes))-1
    for i, (X, y) in enumerate(data):
        y_hat = model(X)
        y_hat = F.softmax(y_hat,dim=1)
        predict[y,:] = y_hat.detach().numpy()
    # For deletes that are all -1
    predict = predict[predict[:,0]>-1]
    return predict
def score(ftra,ftes,label,K,model,Ctra,Ctes,Classic = True):
    Second_Layer_input = np.zeros((Ctes,K,Ctra))
    if Classic:
        from scipy.spatial.distance import cdist
        # Jaccard
        Second_Layer_input[:,0,:] = 1 - cdist(np.array(ftes, dtype=bool), np.array(ftra, dtype=bool), 'jaccard')
        # KL
        Second_Layer_input[:,1,:] = np.square(cdist(ftes,ftra,'jensenshannon'))*2
        # JKL
        Second_Layer_input[:,2,:] = Second_Layer_input[:,0,:]/(Second_Layer_input[:,1,:]+1e-6)
        # KL need to change
        Second_Layer_input[:,1,:] = 2 - Second_Layer_input[:,1,:]
    # NN
    from Core.TrainSetAndTestSet import NN_Input
    Input = DataLoader(NN_Input(ftes, label), batch_size=1, shuffle=False)
    Second_Layer_input[:,K-1,:] = Test_Graph(model,Input,Ctra)
    return Second_Layer_input
def GetValAccs(Feature_train1,label_train1,ValidFeature,model,Classes,date):
    K = 4
    interval = np.unique(date)
    Acc = np.zeros((2,len(interval),K))
    I = 0
    # import time
    # start = time.time()
    for d in interval:
        idx = np.where(date!=d)
        temp = pd.DataFrame(ValidFeature[idx])
        temp2 = temp.groupby(0).sum()
        Feature_train2 = temp2.values
        label_train2 = np.asarray(temp2.index)
        # Similarity and Neural Network Calculation Results
        sim = score(Feature_train1,Feature_train2,label_train2,K,model,Ctra=Classes,Ctes=len(label_train2))
        # Calculate match accuracy
        Acc[0,I,:] = AssignWeights(sim,label_train1,label_train2,Graph=False)
        Acc[1,I,:] = AssignWeights(sim,label_train1,label_train2,Graph=True)
        I= I+1
    # print(time.time()-start)
    return Acc
def Fusion_dateGroup(cfg,userID):
    cfg.Num = len(userID)
    dataname = cfg.data_name[:-4]
    # savename = dataname \
    #            +'U'+str(cfg.Num)+'seed'+str(cfg.seed)+'n'+str(cfg.nGram)+'r'+str(cfg.ratio)+'splitdate'+str(cfg.date[-2])
    Feature_train1,label_train1, \
    FeatureMap2,label2,\
    X,Y, \
    FeatureName,Classes, \
    date,ValidFeature \
        = GetFeature(cfg,userID)
    Inputsize = len(FeatureName)
    methods = ['Jaccard', 'KL', 'JKL', 'CNN']
    K = len(methods)
    loadname = 'U'+str(cfg.Num)+'N'+str(cfg.nGram)+'R'+str(cfg.ratio)+'seed'+str(cfg.seed)
    # name = savepath+dataname+'\\'+loadname
    # Train the model with the training set
    from Core.TrainAndTest import Learner
    model = Learner(FeatureMap=Feature_train1,label=label_train1,
                    FeatureNum=Inputsize,Classes=Classes,cfg=cfg,mixup=True,modelname='CNN')
    torch.save(model.state_dict(),savepath+'models\\'+dataname+'\\'+loadname+'ini_para.pth')
    # Calculate the recognition accuracy on the validation set
    # interval = range(cfg.date[1]+1,cfg.date[2]+1)
    Acc = GetValAccs(Feature_train1, label_train1, ValidFeature, model, Classes, date)
    # Store weight matrix - for subsequent ablation experiments
    np.save(savepath+'weights\\'+dataname+'\\'+dataname+loadname,Acc)
    #  Take the average to get the weight of each method
    weights_naive = np.mean(Acc[0,:,:],axis = 0)
    weights_graph = np.mean(Acc[1,:,:],axis = 0)
    #  Train the model with training and validation data
    model = Learner(np.r_[Feature_train1,FeatureMap2], np.r_[label_train1,label2],
                    FeatureNum=Inputsize,Classes=Classes,cfg=cfg,mixup=True,modelname='CNN')
    torch.save(model.state_dict(),savepath+'models\\'+dataname+'\\'+loadname+"Endpara.pth")
    from Core.Fusion import Score
    sim = Score(Feature_train1+FeatureMap2,X,Y,K,model,Classes,Classic=True)
    np.save(savepath+'sims\\'+dataname+'\\'+loadname+"Score.npy",sim)
    # Identification accuracy of individual methods and fusions on the test set
    acc = np.zeros((2,K+1))
    acc[:,:K] = SingleMatcherAcc(sim,label_train1,Y)
    acc[:,K:(K+1)] = fusionAcc(sim,label_train1,Y,weights_naive,weights_graph)
    return acc