from Core.FusionModel import Fusion_dateGroup
import numpy as np
from Core.TrainSetAndTestSet import RandomlySelectUser
def R(cfg):
    userID = RandomlySelectUser(cfg)
    return userID

def main(cfg,Users,name):
    Acc = np.zeros((len(Users),10,2,5)) # U, seed,NM/GM, [Jac,KL,JKL,CNN,fusion]
    I = 0
    import time
    start = time.time()
    for u in Users:
        cfg.Num = u
        J = 0
        for s in range(10):
            cfg.seed = s
            userID = R(cfg)
            Acc[I,J,:,:] = Fusion_dateGroup(cfg,userID)
            J +=1
        np.save(cfg.savepath+'accs\\'+name,Acc)
        I +=1
        print(time.time()-start)

    Acc = np.load(cfg.savepath+'accs\\'+name,allow_pickle=True)
    print('JKL-NM',np.mean(Acc[:,:,0,2],axis = 1))
    print('JKL-GM',np.mean(Acc[:,:,1,2],axis = 1))
    print('CNN-NM',np.mean(Acc[:,:,0,3],axis = 1))
    print('CNN-GM',np.mean(Acc[:,:,1,3],axis = 1))
    print('fusion-NM',np.mean(Acc[:,:,0,-1],axis = 1))
    print('fusion-GM',np.mean(Acc[:,:,1,-1],axis = 1))

if __name__ == '__main__':

    from Core.utils import config
    cfg = config()
    cfg.date=[0,10,20,31]
    cfg.nGram = 1
    cfg.ratio = 1

    # IPTV
    cfg.ph = 3
    cfg.epochs = 100
    cfg.Num = 1000
    cfg.data_name = "IPTV.txt"
    name = 'IPTVFusionN1R1dateGroup.npy'
    Users = range(100,1000,100)
    main(cfg,Users,name)


    # Shop
    cfg.ph = 4
    cfg.epochs = 200
    cfg.Num = 299
    cfg.data_name = "Shop.txt"
    name = 'ShopFusionN1R1dateGroup.npy'
    Users = range(50,300,50)
    main(cfg,Users,name)


    # Reddit
    cfg.ph = 4
    cfg.epochs = 200
    cfg.Num = 945
    cfg.data_name = "Reddit.txt"
    name = 'RedditFusionN1R1dateGroup.npy'
    Users = range(100,1000,100)
    main(cfg,Users,name)


    # IPTV
    cfg.ph = 3
    cfg.nGram = 2
    cfg.ratio = 0.2
    cfg.epochs = 100
    cfg.Num = 1000
    cfg.data_name = "IPTV.txt"
    name = 'IPTVFusionN2R0.2dateGroup.npy'
    Users = range(100,1000,100)
    main(cfg,Users,name)
