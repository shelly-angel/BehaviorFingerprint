import numpy as np
import time
def UMIS(tra,tes,cfg,method='JS'):
    # Line 1: Time consumed by feature extraction,
    # Line 2: Calculate the time consumed by similarity,
    # Line 3: time consumed by nai matching,
    # Line 4: time consumed by gra matching;
    # The three columns represent the three similarities (Jaccard,JS(KL),JKL) respectively
    Time = np.zeros((4,3))
    # The first row represents the recognition accuracy of simple matching (NM)
    # The second row represents the recognition accuracy of graph matching (GM)
    # The three columns represent the three similarities (Jaccard,JS(KL),JKL) respectively
    Acc = np.zeros((2,3))
    from Core.TrainSetAndTestSet import PrePareData
    from scipy.spatial.distance import cdist
    from Core.utils import MatchID
    Feature_train,label_train,Feature_test,label_test,t = PrePareData(tra,tes,cfg)
    if method=='Jac':
        #(1) Jaccard
        idx = 0
        Time[0,idx] = t
        start = time.perf_counter()
        score = 1 - cdist(np.array(Feature_test, dtype=bool), np.array(Feature_train, dtype=bool), 'jaccard')
        Time[1,idx] = time.perf_counter()-start
    elif method=='JS':
        # (2)KL
        idx = 1
        Time[0,idx] = t
        start = time.perf_counter()
        score = 2 - np.square(cdist(Feature_test,Feature_train,'jensenshannon'))*2
        Time[1,idx] = time.perf_counter()-start
    elif method=='JKL':
        # (3)JKL
        idx = 2
        Time[0,idx] = t
        start = time.perf_counter()
        Jaccard = 1 - cdist(np.array(Feature_test, dtype=bool), np.array(Feature_train, dtype=bool), 'jaccard')
        KL = np.square(cdist(Feature_test,Feature_train,'jensenshannon'))*2
        score = Jaccard/(KL+1e-6)
        Time[1,idx] = time.perf_counter()-start
    elif method=="All":
        Time[0,:] +=t
        # Jaccard
        start = time.perf_counter()
        Jaccard = 1 - cdist(np.array(Feature_test, dtype=bool), np.array(Feature_train, dtype=bool), 'jaccard')
        Time[1,0] = time.perf_counter()-start
        #JS
        start = time.perf_counter()
        KL = np.square(cdist(Feature_test,Feature_train,'jensenshannon'))*2
        Time[1,1] = time.perf_counter()-start
        #JKL
        start = time.perf_counter()
        score = Jaccard/(KL+1e-6)
        Time[1,2] = Time[1,0]+Time[1,1]+time.perf_counter()-start
        # Matching
        pre_ind_nai, pre_ind_gra,Time[2,0],Time[3,0]= MatchID(Jaccard)
        Acc[0,0] = sum(label_test == label_train[pre_ind_nai]) / len(label_test)
        Acc[1,0] = sum(label_test[pre_ind_gra[0]] == label_train[pre_ind_gra[1]]) / len(label_test)
        pre_ind_nai, pre_ind_gra,Time[2,1],Time[3,1]= MatchID(2-KL)
        Acc[0,1] = sum(label_test == label_train[pre_ind_nai]) / len(label_test)
        Acc[1,1] = sum(label_test[pre_ind_gra[0]] == label_train[pre_ind_gra[1]]) / len(label_test)
        pre_ind_nai, pre_ind_gra,Time[2,2],Time[3,2]= MatchID(score)
        Acc[0,2] = sum(label_test == label_train[pre_ind_nai]) / len(label_test)
        Acc[1,2] = sum(label_test[pre_ind_gra[0]] == label_train[pre_ind_gra[1]]) / len(label_test)
        return Acc,Time
    else:
        print('similarity error,please define sim function!')
    # Classification
    from Core.utils import MatchID
    pre_ind_nai, pre_ind_gra,Time[2,idx],Time[3,idx]= MatchID(score)
    Acc[0,idx] = sum(label_test == label_train[pre_ind_nai]) / len(label_test)
    Acc[1,idx] = sum(label_test[pre_ind_gra[0]] == label_train[pre_ind_gra[1]]) / len(label_test)
    return Acc,Time

def Ex3(cfg,x=range(1000,13000,1000),method="JS"):
    subfile = cfg.data_name[:-4]
    savename =  subfile+'\\'+subfile\
               +'U'+str(x[0])+'toU'+str(x[-1]) \
               +'n'+str(cfg.nGram)+'r'+str(cfg.ratio)+'splitdate'+str(cfg.date[-2])
    from Core.DataProcessing import CommonUser,TwoFoldData
    user = CommonUser(data_path=cfg.data_path, data_name=cfg.data_name,date=cfg.date)
    UMISAcc = np.zeros((2,3,len(x)))
    UMISTime = np.zeros((4,3,len(x)))
    for I in range(len(x)):
        cfg.Num = x[I]
        userID = user[:cfg.Num]
        TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,
                                             data_name=cfg.data_name,date=cfg.date)
        UMISAcc[:,:,I],UMISTime[:,:,I] = UMIS(TrainData,TestData,cfg,method=method)
        np.save(cfg.savepath+'accs\\'+savename+"Acc.npy",UMISAcc)
        np.save(cfg.savepath+'times\\'+savename+"Time.npy",UMISTime)
    Acc = {'format':'[NM/GM,JS,UserNum]','Acc':UMISAcc[:,1,:],'Time':UMISTime[:,1,:]}
    return Acc
if __name__ == '__main__':
    from Core.utils import config
    cfg = config()
    # Ex3-More User
    cfg.nGram=2
    cfg.ratio = 0.2
    Acc=Ex3(cfg,x=range(1000,13000,1000)) # Under the circumstance that it is guaranteed to be closet, there are only more than 12,000 IPTV users in total.
    print(Acc)