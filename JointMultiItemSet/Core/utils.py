import numpy as np
from scipy.optimize import linear_sum_assignment
def config():
    import argparse
    arg = argparse.ArgumentParser()
    # data processing
    arg.add_argument('--Num', default=1000, help='the number of users')
    arg.add_argument('--date', default= [0,16,31])
    arg.add_argument('--data_path', default=  "../RawData/", help='the path of data')
    arg.add_argument('--data_name', default= "IPTV.txt", help='the name of data')
    arg.add_argument('--nGram', default= 1, help='MultiSet Feature Para-n')
    arg.add_argument('--ph', default= 4, help='place holder of Max Item')
    arg.add_argument('--ratio', default= 1, help='MultiSet Feature Para-r_{top}')
    arg.add_argument('--mixup', default= True)
    arg.add_argument('--alpha', default=1., type=float,help='mixup interpolation coefficient (default: 1)')
    # Model training settings
    arg.add_argument('--seed', default=0, help='A random seed')
    # arg.add_argument('--HaveModel', default= False, help='Whether to leverage existing models')
    arg.add_argument('--batch_size', default= 1000)
    arg.add_argument('--epochs', default=100,help = "the epoch of training")
    arg.add_argument('--learning_rate', default=0.1,help = "the learning rate of training(SGD=0.1,Adam = 0.0005)")
    arg.add_argument('--momentum', default=0)
    arg.add_argument('--weight_decay', default=0.001,help="L2 regularization coefficients")
    # Data final storage location
    arg.add_argument('--savepath', default="../Result/")
    # parameter parsing
    cfg = arg.parse_args()
    return cfg
def MatchID(Sim,value=0.0,sim = True):
    import time
    np.nan_to_num(Sim,nan = value,posinf=value,neginf=value,copy=False) #If there are some nan values, replace them all with 0
    if sim:
        t1 = time.perf_counter()
        pre_ind_nai = Sim.argmax(axis=1) #NM
        t1 = (time.perf_counter()-t1)
    else:
        t1 = time.perf_counter()
        pre_ind_nai = Sim.argmin(axis=1)
        t1 = (time.perf_counter()-t1)
    t2 = time.perf_counter()
    pre_ind_gra = linear_sum_assignment(Sim, maximize=sim) #GM
    t2 = (time.perf_counter()-t2)
    return pre_ind_nai,pre_ind_gra,t1,t2