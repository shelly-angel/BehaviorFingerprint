import numpy as np
from scipy.optimize import linear_sum_assignment

def Fusion_SumRule(weights,Score,Matching = 'NM'):
    weights = weights/np.sum(weights)
    K = len(weights)
    Classes = Score.shape[0]
    temp = np.zeros((Classes, Classes))
    for I in range(K):
        temp = weights[I] * Score[:, I, :] + temp
    np.nan_to_num(temp, copy=False)
    if Matching=='NM':
        Y = np.arange(Classes)
        predict_label_index = temp.argmax(axis=1)
        Acc = sum(Y == predict_label_index) / len(Y)
    elif Matching=='GM':
        predict_label_index = linear_sum_assignment(temp, maximize=True)
        Acc = sum(predict_label_index[0] == predict_label_index[1]) / len(predict_label_index[0])
    return Acc
def ablation(path,dataname,Classes,nGram,ratio,seed):
    loadname = "U"+str(Classes)+"N"+str(nGram)+"R"+str(ratio)+"seed"+str(seed)
    # load weights
    weights = np.load(path+'weights\\'+dataname+'\\'+dataname+loadname+'.npy',allow_pickle=True)
    weights_naive = np.mean(weights[0,:,:],axis = 0)
    weights_graph = np.mean(weights[1,:,:],axis = 0)
    # Load similarity calculation results
    Score = np.load(path+'sims\\'+dataname+'\\'+"U"+str(Classes)+"N"+str(nGram)+"R"+str(ratio)+'seed0'
                    +'Score.npy',allow_pickle=True)
    if seed!=0:
        a = np.load(path+'sims\\'+dataname+'\\'+loadname+'Score.npy',allow_pickle=True) # 每一次只有NN不一样
        Score[:,-1,:] = a
    # normalize the score
    Score[:,1,:] = Score[:,1,:]/2
    JKL_min = np.min(Score[:,2,:]);JKL_max=np.max(Score[:,2,:])
    Score[:,2,:] = (Score[:,2,:] -JKL_min)/(JKL_max-JKL_min)
    # fusion
    M = 5
    Acc = np.zeros((2,M)) #Fusion-Sum,Fusion-Jaccard,F-KL,F-JKL,F-NN
    #Sum-rule
    Acc[0,0] = Fusion_SumRule(weights_naive,Score,'NM')
    Acc[1,0] = Fusion_SumRule(weights_graph,Score,'GM')
    for J in range(M-1):
        weight = np.delete(weights_naive,J,axis=0)
        proba = np.delete(Score,J,axis=1)
        Acc[0,J+1] = Fusion_SumRule(weight,proba,'NM')
        weight = np.delete(weights_graph,J,axis=0)
        Acc[1,J+1] = Fusion_SumRule(weight,proba,'GM')
    return Acc

def ex(path,dataname,Classes,nGram=1,ratio=1):
    Acc = np.zeros((10,2,5))
    for seed in range(10):
        Acc[seed,:,:] = ablation(path,dataname,Classes,nGram,ratio,seed)
    print(np.mean(Acc,axis=0))
    return Acc

from Core.utils import config
cfg = config()
path = cfg.savepath
Acc = np.zeros((4,10,2,5))
#IPTV
Acc[0,:,:,:] = ex(path,dataname='IPTV',Classes=1000,nGram=1,ratio=1)
#Shop
Acc[1,:,:,:] = ex(path,dataname='Shop',Classes=299,nGram=1,ratio=1)
#Reddit
Acc[2,:,:,:] = ex(path,dataname='Reddit',Classes=945,nGram=1,ratio=1)
#IPTV
Acc[3,:,:,:] = ex(path,dataname='IPTV',Classes=1000,nGram=2,ratio=0.2)