# path = 'F:\\UserIdentification\\UserIdentification_Seq\\RawData\\'
# filename = 'IPTV_SortedByUsernum.txt'

import pandas as pd
import numpy as np

def config():
    import argparse
    arg = argparse.ArgumentParser()
    # 数据处理
    arg.add_argument('--data_dir',
                     default= "D:\\Research\\UserIdentification\\Codes\\Ours\\RawData\\",
                     help='the path of data ')
    arg.add_argument('--filename',
                     default= "IPTV.txt",
                     help='the name of data')
    arg.add_argument('--date', default=[(0,20),(21,31)],
                     help='[(tra_date1,tra_enddate),(val_date1,val_enddate)]')
    arg.add_argument('--date_ClosedSet',default=[(0,10),(11,20),(21,31)])
    arg.add_argument('--period',default=3600)
    arg.add_argument('--item_size', default=2000,
                     help='词典大小')
    arg.add_argument('--emb_len',default=32)
    arg.add_argument('--max_seq_length',default=300,
                     help='序列长度')
    ## 网络设置
    arg.add_argument('--num_atten_heads',default=4,
                     help='多头注意力机制有多少个头')
    arg.add_argument('--num_encoder_layers',default=2,
                     help='Encoder里面有几层')
    arg.add_argument('--atten_dp_prob',default=0.5,
                     help='self_atten 后面有一个dropout')
    arg.add_argument('--dp_prob',default=0.5,
                     help='feedforward 部分的dropout')
    arg.add_argument('--hidden_act',default='gelu',
                     help='feedforward 中全连接网络的激活函数')
    arg.add_argument('--hidden_size',default=64*4,
                     help='feedforward 中全连接网络的中间层大小')
    arg.add_argument("--initializer_range", type=float, default=0.02)
    # 训练参数
    # train args
    arg.add_argument("--lr", type=float, default=0.00001, help="learning rate of adam")
    arg.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    arg.add_argument("--epochs", type=int, default=200, help="number of epochs")
    arg.add_argument("--cuda_condition",default=False)
    # arg.add_argument("--no_cuda", action="store_true")
    # arg.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    arg.add_argument("--seed", default=42, type=int)
    arg.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    arg.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    arg.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    # arg.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # 参数解析
    cfg = arg.parse_args()
    return cfg

class DataProcessing:
    def __init__(self,args):
        self.data_dir = args.data_dir
        self.filename = args.filename
        self.datapath = args.data_dir + args.filename
        self.savename = args.data_dir+args.filename[:-4]+'_seq'+str(args.period/3600)+'.txt'
        self.date = args.date
        if args.date_ClosedSet is None:
            self.date_ClosedSet = self.date
            self.userID = range(1,1001)
        else:
            self.date_ClosedSet = args.date_ClosedSet
        self.period = args.period

    def Commonuser(self):
        print(f"{'提取公共用户:'}{self.date_ClosedSet}")
        data = pd.read_csv(self.datapath,sep='\t',
                           dtype=int,names=['id', 'Item', 'time']);  # index_col=0,that means the index is the value of first column
        data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int') #第几天
        d = len(self.date_ClosedSet)
        for i in range(d):
            a,b = self.date_ClosedSet[i][0],self.date_ClosedSet[i][1]
            tmp_u = data[(data['date'] >= a) & (data['date'] <= b)]['id'].values
            if i==0:
                user = tmp_u
            else:
                user = np.intersect1d(tmp_u,user)
        return user
    def dataset(self,userID=None):
        # 导入数据
        print('从文件导入数据'+self.filename)
        if userID is None:
            userID = self.userID
        data = pd.read_csv(self.datapath,sep='\t',
                           dtype=int, names=['id', 'Item', 'time']);  # index_col=0,that means the index is the value of first column
        data.index = data['id']
        data = data.loc[userID]
        data['id'] = pd.Categorical(data['id']).codes + 1  # userID 如果不连续的话; 三个数据集的Item都是从1开始的。
        data = data.reset_index(drop=True)
        # data['Item'] = pd.Categorical(data['Item']).codes+1   # 用户减少了，频道编号可能不连续了,重新编号使其连续，减少空间浪费, 从1开始，0要用来补空位padding
        data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int') #第几天
        #划分数据
        TrainData = data[(data['date'] >= self.date[0][0]) & (data['date'] <= self.date[0][1])]
        ValidData = data[(data['date'] >= self.date[1][0]) & (data['date'] <= self.date[1][1])]
        TrainData.reset_index(drop=True)
        ValidData.reset_index(drop=True)
        return TrainData,ValidData
    def sequential(self,data,period =None,save=False):
        if period is None:
            period = self.period
        data.time = data.time-data.date*(24*3600)
        data.time = np.floor(data.time/period).astype('int') # dura=3600s,则date.time 表示一天中的几点
        behavSeq_grp = data.groupby(["id","date","time"]).agg(
            behav_type=("Item",list)
        ).reset_index()
        if save:
            behavSeq_grp.to_csv(self.savename,sep='\t',index=False, header=None)
        return behavSeq_grp


if __name__ == '__main__':

    cfg = config()
    data = DataProcessing(cfg)
    user = data.Commonuser()
    tra,val = data.dataset(user[:100])
    tra = data.sequential(tra)
    val = data.sequential(val)
