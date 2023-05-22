from Core.FusionModel import Fusion_dateGroup
def All(cfg):
    from Core.setRandomSeed import set_random_seed
    set_random_seed(cfg.seed)
    print('Fixed random seed')
    from Core.DataProcessing import Alluser
    userID = Alluser(data_path=cfg.data_path,data_name=cfg.data_name)
    return userID
if __name__ == '__main__':
    import time
    from Core.utils import config
    cfg = config()
    cfg.date=[0,10,20,31]
    cfg.nGram = 1
    cfg.ratio = 1

    # IPTV
    cfg.ph = 3
    cfg.epochs = 100
    cfg.data_name = "IPTV.txt"
    savename = 'IPTVFusionU1000seed0n1r1splitdate20'
    cfg.seed = 0
    start = time.time()
    userID = All(cfg)
    acc=Fusion_dateGroup(cfg,userID)
    print(time.time()-start)
    print(acc)

    # Shop
    cfg.ph = 4
    cfg.epochs = 200
    cfg.data_name = "Shop.txt" # IPTV之前存储了模型，所以一个小时
    savename = 'ShopFusionU299seed0n1r1splitdate20'
    cfg.seed = 0
    start = time.time()
    userID = All(cfg)
    acc=Fusion_dateGroup(cfg,userID)
    print(time.time()-start)
    print(acc)

    # Reddit
    cfg.ph = 4
    cfg.epochs = 200
    cfg.Num = 945
    cfg.data_name = "Reddit.txt"
    savename = 'RedditFusionU945seed0n1r1splitdate20'
    cfg.seed = 0
    start = time.time()
    userID = All(cfg)
    acc=Fusion_dateGroup(cfg,userID)
    print(time.time()-start)
    print(acc)

    # IPTV
    cfg.ph = 3
    cfg.nGram = 2
    cfg.ratio = 0.2
    cfg.epochs = 100
    cfg.Num = 1000
    cfg.data_name = "IPTV.txt" # IPTV之前存储了模型，所以一个小时
    savename = 'IPTVFusionU1000seed0n2r0.2splitdate20'
    cfg.seed = 0
    start = time.time()
    userID = All(cfg)
    acc=Fusion_dateGroup(cfg,userID)
    print(time.time()-start)
    print(acc)