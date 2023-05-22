import numpy as np
from Core.utils import config
cfg = config()
path = cfg.savepath

def ACC_K(score,knownuser,unknownuser,K=1):
    rnk = np.argsort(-score,axis=1)
    predict_label = knownuser[rnk[:,:K]]
    cnt = np.asarray([1<0 for _ in range(len(unknownuser))])
    for J in range(K):
        cnt = cnt|(predict_label[:,J]==unknownuser)
    return sum(cnt)/len(unknownuser)

def res(score,label_train,label_test):
    a = []
    for k in [1,2,3,4,5]: # range(1,20+1)
        a.append(round(ACC_K(score,label_train,label_test,K=k),4))
    print(a)
    return a
def res2(Score,fus,label_train,label_test):
    print('compare')
    a = []
    a.append(res(Score[:,0,:],label_train,label_test)) #Jaccard
    a.append(res(Score[:,1,:],label_train,label_test)) #KL
    a.append(res(Score[:,2,:],label_train,label_test)) #JKL
    a.append(res(Score[:,3,:],label_train,label_test)) #NN
    a.append(res(fus,label_train,label_test))# fusion
    return a

def f(path,dataname,Classes,nGram,ratio,seed):
    ## 加载相似度计算结果
    loadname = "U"+str(Classes)+"N"+str(nGram)+"R"+str(ratio)+"seed"+str(seed)
    Score = np.load(path+'sims\\'+dataname+'\\'+"U"+str(Classes)+"N"+str(nGram)+"R"+str(ratio)+'seed0'
                    +'Score.npy',allow_pickle=True)
    if seed!=0:
        a = np.load(path+'sims\\'+dataname+'\\'+loadname+'Score.npy',allow_pickle=True) # 每一次只有NN不一样
        Score[:,-1,:] = a
    label_train = np.arange(Classes)
    label_test = np.arange(Classes)
    ## 计算融合的结果
    # 加载权重
    weights = np.load(path+'weights\\'+dataname+'\\'+dataname+loadname+'.npy',allow_pickle=True)
    weights_naive = np.mean(weights[0,:,:],axis = 0)
    weights_graph = np.mean(weights[1,:,:],axis = 0)
    # 给分数归一化
    Score[:,1,:] = Score[:,1,:]/2
    JKL_min = np.min(Score[:,2,:]);JKL_max=np.max(Score[:,2,:])
    Score[:,2,:] = (Score[:,2,:] -JKL_min)/(JKL_max-JKL_min)
    # 计算融合后的分数
    weights = weights_naive/np.sum(weights_naive)
    # weights = weights/np.sum(weights)
    K = len(weights)
    Classes = Score.shape[0]
    temp = np.zeros((Classes, Classes))
    for I in range(K):
        temp = weights[I] * Score[:, I, :] + temp
    np.nan_to_num(temp, copy=False)
    a = res2(Score,temp,label_train,label_test)
    print(a)
    return a

a = f(path,dataname='IPTV',Classes=1000,nGram=1,ratio=1,seed=0)
b = f(path,dataname='Shop',Classes=299,nGram=1,ratio=1,seed=0)
c = f(path,dataname='Reddit',Classes=945,nGram=1,ratio=1,seed=0)
