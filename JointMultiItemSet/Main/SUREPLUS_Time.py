import numpy as np
from Main.UnionVSInter_Acc import Uni,Inter
from Main.SUREPLUS import UMIS

def Ex2(cfg):
    x = range(1000,13000,1000)
    subfile = cfg.data_name[:-4]
    savename = cfg.savepath+'times\\'+ str(subfile)+'\\'+subfile \
               +'U'+str(x[0])+'Group'+str(len(x)) \
               +'n'+str(cfg.nGram)+'r'+str(cfg.ratio)+'splitdate'+str(cfg.date[-2])
    from Core.DataProcessing import CommonUser,TwoFoldData
    user = CommonUser(data_path=cfg.data_path, data_name=cfg.data_name,date=cfg.date)
    InterTime = np.zeros((4,3,len(x)))
    UnionTime = np.zeros((4,3,len(x)))
    UMISTime = np.zeros((4,3,len(x)))
    for I in range(len(x)):
        cfg.Num = x[I]
        if I>0:
            # Time comparison needs to be done,
            # which is the average of the time spent by multiple groups of equal numbers of users
            userID = user[x[I-1]:cfg.Num]
        else:
            userID = user[:cfg.Num]

        TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,
                                             data_name=cfg.data_name,date=cfg.date)
        _,InterTime[:,:,I] = Inter(TrainData,TestData,cfg)
        np.save(savename+"Inter.npy",InterTime)
        _,UnionTime[:,:,I] = Uni(TrainData,TestData,cfg)
        np.save(savename+"Union.npy",InterTime)
        _,UMISTime[:,:,I] = UMIS(TrainData,TestData,cfg,method="All")
        np.save(savename+"UnionAll.npy",UMISTime)
    def output(tt):
        A = tt[:3].sum(axis=0)
        B = tt[[0,1,3]].sum(axis=0)
        return A.mean(axis=1),B.mean(axis=1)
    print('nGram=%d,ratio=%.1f'%(cfg.nGram,cfg.ratio))
    print('format=[[NM-Jac,NM-JS,NM-JKL],[GM-Jac,JM-JS,GM-JKL]]')
    print('InterTimeMean=',output(InterTime))
    print('UnionTimeMean=',output(UnionTime))
    print('UnionAllTimeMean=',output(UMISTime))

if __name__ == '__main__':

    from Core.utils import config
    cfg = config()
    # Ex2-Time of (Inter VS Union VS UnionAll)
    cfg.nGram = 1
    cfg.ratio = 1
    Ex2(cfg)
    cfg.nGram = 2
    cfg.ratio = 1
    Ex2(cfg)