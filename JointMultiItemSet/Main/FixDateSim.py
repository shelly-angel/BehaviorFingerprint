import pandas as pd
import numpy as np
from Main.SUREPLUS import UMIS
def TrainUser(data_path,data_name ='IPTV.txt',date = [0,16,31]):
    data = pd.read_csv(data_path + data_name, sep='\t', dtype=int, names=['id', 'Item', 'time']);  # index_col=0,that means the index is the value of first column
    data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int')
    TrainData = data[(data['date'] >= date[0]) & (data['date'] <= date[1])]
    user = np.unique(TrainData['id'])
    return user
def TwoFoldData(userID=range(1,1001),data_path = " ",data_name ='IPTV.txt',date = [0,16,31]):
    print('1.Import data from the file:'+data_name) #
    data = pd.read_csv(data_path + data_name,sep='\t', dtype=int, names=['id', 'Item', 'time'])
    data.index = data['id']
    data = data.loc[userID]
    data['id'] = pd.Categorical(data['id']).codes + 1
    data = data.reset_index(drop=True)
    data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int')
    data = data.drop(['time'],axis = 1)
    data['Item'] = pd.Categorical(data['Item']).codes + 1
    print("2.Divide the dataset by date(train,test)=(%d~%d,%d~%d)"%(date[0],date[1],date[1]+1,date[2]))
    # Divide data
    TrainData = data[(data['date'] >= date[0]) & (data['date'] <= date[1])]
    ValidData = data[(data['date'] > date[1]) & (data['date'] <= date[2])]
    Classes = len(userID)
    return TrainData,ValidData,Classes
def FixKnown(cfg,split_date=4):
    subfile = cfg.data_name[:-4]
    savename = subfile +'\\'+subfile \
               +'U'+str(cfg.Num) \
               +'n'+str(cfg.nGram)+'r'+str(cfg.ratio)+'FixknownDate'+str(split_date)
    # from DataProcessing import TwoFoldData
    # userID = range(1,cfg.Num+1,1)
    user = TrainUser(data_name=cfg.data_name,date=[0,split_date,31])
    if cfg.Num>len(user):
        cfg.Num = len(user)
    userID = user[:cfg.Num]
    x = [i for i in range(split_date+1,32)]
    UMISAcc = np.zeros((2,3,len(x)))
    UMISTime = np.zeros((4,3,len(x)))
    for I in range(len(x)):
        cfg.date = [0,split_date,x[I]]
        TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,
                                             data_name=cfg.data_name,date=cfg.date)
        UMISAcc[:,:,I],UMISTime[:,:,I] = UMIS(TrainData,TestData,cfg,method='All')
        np.save(cfg.savepath+'accs\\'+savename+"Acc.npy",UMISAcc)
        np.save(cfg.savepath+'times\\'+savename+"Time.npy",UMISTime)
    print('n=%d,r=%.2f'%(cfg.nGram,cfg.ratio))
    print('FixKnown,Jaccard-NM=',UMISAcc[0,0,:])
    return UMISAcc
def FixUnknown(cfg,start_date=26):
    subfile = cfg.data_name[:-4]
    savename = subfile +'\\'+subfile \
               +'U'+str(cfg.Num) \
               +'n'+str(cfg.nGram)+'r'+str(cfg.ratio)+'FixunknownDate'+str(start_date+1)
    # from DataProcessing import TwoFoldData
    userID = range(1,cfg.Num+1,1)
    x = [i for i in range(start_date,-1,-1)]
    UMISAcc = np.zeros((2,3,len(x)))
    UMISTime = np.zeros((4,3,len(x)))
    for I in range(len(x)):
        cfg.date = [x[I],start_date,31]
        TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,
                                             data_name=cfg.data_name,date=cfg.date)
        UMISAcc[:,:,I],UMISTime[:,:,I] = UMIS(TrainData,TestData,cfg,method='All')
        np.save(cfg.savepath+'accs\\'+savename+"Acc.npy",UMISAcc)
        np.save(cfg.savepath+'times\\'+savename+"Time.npy",UMISTime)
    print('n=%d,r=%.2f'%(cfg.nGram,cfg.ratio))
    print('FixunKnown,Jaccard-NM=',UMISAcc[0,0,:])
    return UMISAcc
if __name__ == '__main__':
    from Core.utils import config
    cfg = config()
    cfg.nGram =1
    cfg.ratio=1
    cfg.data_name = 'Shop.txt'
    cfg.ph=4
    cfg.epochs = 400
    cfg.Num= 300
    FixKnown(split_date=10)
    FixUnknown(start_date=20)