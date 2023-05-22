import numpy as np
import pandas as pd
import time
def Inter(tra,tes,cfg):
    # Line 1: Time consumed by feature extraction,
    # Line 2: Calculate the time consumed by similarity,
    # Line 3: time consumed by nai matching,
    # Line 4: time consumed by gra matching;
    # The three columns represent the three similarities (Jaccard,JS(KL),JKL) respectively
    Time = np.zeros((4,3))
    # The first row represents the recognition accuracy of simple matching (NM)
    # The second row represents the recognition accuracy of graph matching (GM)
    # The three columns represent the three similarities (Jaccard,JS(KL),JKL) respectively
    acc = np.zeros((2,3))
    from Core.Fingerprint import SURE
    Featra = SURE(tra,max_value_len=cfg.ph,Multi=cfg.nGram)
    start = time.perf_counter()
    Feates = SURE(tes,max_value_len=cfg.ph,Multi=cfg.nGram)
    Time[0,:] = time.perf_counter()-start
    # similarity calculation
    knownuser = np.asarray(list(Featra.keys()))
    unknownuser = np.asarray(list(Feates.keys()))
    len_knownuser = len(knownuser)
    len_unknownuser = len(unknownuser)
    Jaccard = np.zeros((len_unknownuser,len_knownuser))
    KL = np.zeros((len_unknownuser,len_knownuser))
    I = 0
    from scipy.spatial import distance
    from Core.utils import MatchID
    start = time.perf_counter()
    for u in unknownuser:
        fea_tesu = Feates[u]
        feature_u_test_index = fea_tesu.index
        J = 0
        for v in knownuser:
            fea_trav = Featra[v]
            feature_u_train_index = fea_trav.index
            # The intersection feature is used
            inter_fea = np.intersect1d(feature_u_test_index,feature_u_train_index)
            # Jaccard
            s = time.perf_counter()
            union_fea = np.union1d(feature_u_test_index,feature_u_train_index)
            Jaccard[I,J] = len(inter_fea)/len(union_fea)
            Time[1,0]  += time.perf_counter()-s
            # JS
            s = time.perf_counter()
            feature_u_test = fea_tesu.loc[inter_fea] # The extracted order is consistent with inter_fea
            feature_u_train = fea_trav.loc[inter_fea]
            feature_u_test = list(feature_u_test['times'].values) #/ np.sum(feature_u_test['times'].values)
            feature_u_train = list(feature_u_train['times'].values) #/ np.sum(feature_u_train['times'].values)
            KL[I,J] = np.square(distance.jensenshannon(feature_u_test,feature_u_train))*2
            Time[1,1]  += time.perf_counter()-s
            J = J+1
        I = I+1
    T_Sim = time.perf_counter()-start - Time[1,0]-Time[1,1]
    s = time.perf_counter()
    JKL = Jaccard/(KL+1e-6)
    Time[1,2]  = time.perf_counter()-s + Time[1,0] + Time[1,1]
    Time[1,:] = Time[1,:]+T_Sim
    # # matching users
    #1.Jaccard
    pre_ind_nai, pre_ind_gra ,Time[2,0],Time[3,0]= MatchID(Jaccard)
    acc[0,0] = sum(unknownuser == knownuser[pre_ind_nai]) / len_unknownuser
    acc[1,0] = sum(unknownuser[pre_ind_gra[0]] == knownuser[pre_ind_gra[1]]) / len_unknownuser
    #2.JS
    pre_ind_nai, pre_ind_gra ,Time[2,1],Time[3,1]= MatchID(2-KL)
    acc[0,1] = sum(unknownuser == knownuser[pre_ind_nai]) / len_unknownuser
    acc[1,1] = sum(unknownuser[pre_ind_gra[0]] == knownuser[pre_ind_gra[1]]) / len_unknownuser
    #3.JKL
    pre_ind_nai, pre_ind_gra ,Time[2,2],Time[3,2]= MatchID(JKL)
    acc[0,2] = sum(unknownuser == knownuser[pre_ind_nai]) / len_unknownuser
    acc[1,2] = sum(unknownuser[pre_ind_gra[0]] == knownuser[pre_ind_gra[1]]) / len_unknownuser
    return acc,Time
def Uni(tra,tes,cfg):
    # Line 1: Time consumed by feature extraction,
    # Line 2: Calculate the time consumed by similarity,
    # Line 3: time consumed by nai matching,
    # Line 4: time consumed by gra matching;
    # The three columns represent the three similarities (Jaccard,JS(KL),JKL) respectively
    Time = np.zeros((4,3))
    # The first row represents the recognition accuracy of simple matching (NM)
    # The second row represents the recognition accuracy of graph matching (GM)
    # The three columns represent the three similarities (Jaccard,JS(KL),JKL) respectively
    acc = np.zeros((2,3))
    from Core.Fingerprint import SURE
    Featra = SURE(tra,max_value_len=cfg.ph,Multi=cfg.nGram)
    start = time.perf_counter()
    Feates = SURE(tes,max_value_len=cfg.ph,Multi=cfg.nGram)
    Time[0,:] = time.perf_counter()-start
    # similarity calculation
    knownuser = np.asarray(list(Featra.keys()))
    unknownuser = np.asarray(list(Feates.keys()))
    len_knownuser = len(knownuser)
    len_unknownuser = len(unknownuser)
    Jaccard = np.zeros((len_unknownuser,len_knownuser))
    KL = np.zeros((len_unknownuser,len_knownuser))
    I = 0
    from scipy.spatial import distance
    from Core.utils import MatchID
    start = time.perf_counter()
    for u in unknownuser:
        fea_tesu = Feates[u]
        feature_u_test_index = fea_tesu.index
        J = 0
        for v in knownuser:
            fea_trav = Featra[v]
            feature_u_train_index = fea_trav.index
            # The union of two user features is used
            union_fea = np.union1d(feature_u_test_index,feature_u_train_index)
            # Jaccard
            s = time.perf_counter()
            inter_fea = np.intersect1d(feature_u_test_index,feature_u_train_index)
            Jaccard[I,J] = len(inter_fea)/len(union_fea)
            Time[1,0] += time.perf_counter()-s
            # JS
            s = time.perf_counter()
            ###### One way (official code change, no longer valid)
            # feature_u_test = fea_tesu.loc[union_fea] # The extracted order is consistent with union_fea, and the one that is not is recorded as NaN----no longer valid
            # feature_u_train = fea_trav.loc[union_fea]
            # feature_u_test.fillna(0,inplace=True) # Where there is no record, it is replaced with 0
            # feature_u_train.fillna(0,inplace=True)
            # # KL[I,J] = np.square(cdist([feature_u_test['times'].values],[feature_u_train['times'].values],'jensenshannon'))*2
            # feature_u_test = list(feature_u_test['times'].values) #/ np.sum(feature_u_test['times'].values)
            # feature_u_train = list(feature_u_train['times'].values) #/ np.sum(feature_u_train['times'].values)
            # KL[I,J] = np.square(distance.jensenshannon(feature_u_test,feature_u_train))*2
            ###### Another Way
            temp = pd.DataFrame(np.zeros((len(union_fea),2),dtype='int32'),index=union_fea,columns=['tra','tes'])
            temp.loc[fea_trav.index,'tra'] = fea_trav.times
            temp.loc[fea_tesu.index,'tes'] = fea_tesu.times
            KL[I,J] = np.square(distance.jensenshannon(list(temp.tes.values),list(temp.tra.values)))*2
            Time[1,1]  += time.perf_counter()-s
            J = J+1
        I = I+1
    T_Sim = time.perf_counter()-start - Time[1,0]-Time[1,1]
    s = time.perf_counter()
    JKL = Jaccard/(KL+1e-6)
    Time[1,2] = time.perf_counter()-s + Time[1,0] + Time[1,1]
    Time[1,:] = Time[1,:]+T_Sim
    # # Matching users
    #1.Jaccard
    pre_ind_nai, pre_ind_gra ,Time[2,0],Time[3,0]= MatchID(Jaccard)
    acc[0,0] = sum(unknownuser == knownuser[pre_ind_nai]) / len_unknownuser
    acc[1,0] = sum(unknownuser[pre_ind_gra[0]] == knownuser[pre_ind_gra[1]]) / len_unknownuser
    #2.KL
    pre_ind_nai, pre_ind_gra ,Time[2,1],Time[3,1]= MatchID(2-KL)
    acc[0,1] = sum(unknownuser == knownuser[pre_ind_nai]) / len_unknownuser
    acc[1,1] = sum(unknownuser[pre_ind_gra[0]] == knownuser[pre_ind_gra[1]]) / len_unknownuser
    #3.JKL
    pre_ind_nai, pre_ind_gra ,Time[2,2],Time[3,2]= MatchID(JKL)
    acc[0,2] = sum(unknownuser == knownuser[pre_ind_nai]) / len_unknownuser
    acc[1,2] = sum(unknownuser[pre_ind_gra[0]] == knownuser[pre_ind_gra[1]]) / len_unknownuser
    return acc,Time
def Ex1(cfg,x=range(100,1100,100)):
    subfile=cfg.data_name[:-4]
    savename = cfg.savepath+'accs\\'+ str(subfile)+'\\'+subfile\
               +'U'+str(x[0])+'toU'+str(x[-1]) \
               +'n'+str(cfg.nGram)+'r'+str(cfg.ratio)+'splitdate'+str(cfg.date[-2])
    from Core.DataProcessing import CommonUser,TwoFoldData
    user = CommonUser(data_path=cfg.data_path, data_name=cfg.data_name,date=cfg.date)
    InterAcc = np.zeros((2,3,len(x)))
    UnionAcc = np.zeros((2,3,len(x)))
    for I in range(len(x)):
        cfg.Num = x[I]
        userID = user[:cfg.Num]
        TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,
                                             data_name=cfg.data_name,date=cfg.date)
        InterAcc[:,:,I],_ = Inter(TrainData,TestData,cfg)
        np.save(savename+"Inter.npy",InterAcc)
        UnionAcc[:,:,I],_ = Uni(TrainData,TestData,cfg)
        np.save(savename+"Union.npy",InterAcc)
    Acc = {'format':'[NM/GM,(Jac/JS/JKL),UserNum]','Inter':InterAcc,'Union':UnionAcc}
    print(Acc)
    return Acc
if __name__ == '__main__':
    from Core.utils import config
    cfg = config()
    # Ex1-Accuracy of Inter VS Union
    cfg.nGram = 1
    cfg.ratio = 1
    Acc = Ex1(cfg,x=range(100,1100,100))
    cfg.nGram = 2
    cfg.ratio = 1
    Acc = Ex1(cfg,x= range(100,1100,100))