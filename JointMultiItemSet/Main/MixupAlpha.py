import numpy as np
from Main.Mixup import NN

def main(cfg):
    alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,2.,3.,5.,10.,20.,30.,50.,100.]
    subfile = cfg.data_name[:-4]
    savename = subfile + '\\' + subfile \
               +'U'+str(cfg.Num)+'n'+str(cfg.nGram)+'r'+str(cfg.ratio) \
               +'splitdate'+str(cfg.date[-2])+'alpha'+str(alpha[0])+'_'+str(alpha[-1])
    from Core.DataProcessing import CommonUser,TwoFoldData
    user = CommonUser(data_path=cfg.data_path, data_name=cfg.data_name,date=cfg.date)
    if len(user)<cfg.Num:
        cfg.Num = len(user)
    userID = user[:cfg.Num]
    TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,data_name=cfg.data_name,date=cfg.date)
    Loss = np.zeros((len(alpha),cfg.epochs,3))
    Acc = np.zeros((len(alpha),cfg.epochs,2,2))
    I = 0
    for a in alpha:
        cfg.alpha = a
        Loss[I,:,:],Acc[I,:,:,:] = NN(TrainData,TestData,cfg,mixup=True,modelname='CNN',savepath=cfg.savepath,savename=savename+'CNNMix')
        print('alph,NM-ACC,GM-Acc=',cfg.alpha,Acc[I,-1,0,1],Acc[I,-1,1,1])
        np.save(cfg.savepath+'loss\\'+savename,Loss)
        np.save(cfg.savepath+'accs\\'+savename,Acc)
        I = I+1

if __name__ == '__main__':
    from Core.utils import config
    cfg = config()
    cfg.data_name = "IPTV.txt"
    cfg.nGram = 1
    cfg.ratio = 1
    cfg.epochs = 150
    main()

    cfg.nGram =1
    cfg.ratio=1
    cfg.data_name = 'Shop.txt'
    cfg.ph=4
    cfg.epochs = 400
    main()


    cfg.nGram=1
    cfg.ratio=1
    cfg.data_name = 'Reddit.txt'
    cfg.ph=4
    cfg.epochs = 400
    main()
