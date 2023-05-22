# Google  Colab的config
def Transconfig():
    import argparse
    arg = argparse.ArgumentParser()
    # 数据处理
    arg.add_argument('--data_dir', default="./IPTV_data/", help='数据路径')
    arg.add_argument('--filename', default="Reddit.txt", help='数据文件名')
    arg.add_argument('--save_dir', default="./ContrastiveLearning/Result/", help='结果存储路径')
    arg.add_argument('--date', default=[(0, 20), (21, 31)],
                     help='[(tra_date1,tra_enddate),(val_date1,val_enddate)]')
    arg.add_argument('--date_ClosedSet', default=[(0, 10), (11, 20), (21, 31)])
    arg.add_argument('--period', default=3600 * 24, help='period s的行为记录构成一个样本')
    arg.add_argument('--item_size', default=3600, help='词典大小')
    arg.add_argument('--emb_len', default=300, help='一个词要变成多少维度的')
    arg.add_argument('--max_seq_length', default=121, help='输入网络的序列的长度最大值')
    arg.add_argument('--users', default=100, help='待识别用户数')
    arg.add_argument('--classes', default=100)
    arg.add_argument('--margin', default=0.1)

    ## 网络设置
    arg.add_argument('--num_atten_heads', default=4, help='多头注意力机制有多少个头')
    arg.add_argument('--atten_head_size', default=32, help='多头注意力机制每个头的size')
    arg.add_argument('--num_encoder_layers', default=2, help='Encoder里面有几层')
    arg.add_argument('--atten_dp_prob', default=0.5, help='self_atten 后面有一个dropout')
    arg.add_argument('--dp_prob', default=0.5, help='feedforward 部分的dropout')
    arg.add_argument('--hidden_act', default='gelu', help='feedforward 中全连接网络的激活函数')
    arg.add_argument('--hidden_size', default=300, help='feedforward 中全连接网络的中间层大小')
    arg.add_argument("--initializer_range", type=float, default=0.02)
    # 训练参数
    # train args
    arg.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    arg.add_argument("--batch_size", type=int, default=500, help="测试时的batch_size,太大会out of memory!")
    arg.add_argument("--epochs", type=int, default=200, help="number of epochs")
    arg.add_argument("--B_samples", type=int, default=5, help="每个类别抽多少样本")
    arg.add_argument("--B_classes", type=int, default=100, help="每个Batch抽多少类别")
    arg.add_argument("--cuda_condition", default=True)
    arg.add_argument("--seed", default=1990605, type=int)
    arg.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    arg.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    arg.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    # arg.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    arg.add_argument("--mask_ratio", type=float, default=0.2, help="adam second beta value")

    # 参数解析
    cfg = arg.parse_args('')  # running in ipynb
    # cfg = arg.parse_args()  # Pycharm的格式
    return cfg

def DPCNNconfig():
    import argparse
    arg = argparse.ArgumentParser()
    # 数据处理
    arg.add_argument('--data_dir',default= "./IPTV_data/",help='数据路径')
    arg.add_argument('--filename',default= "IPTV.txt",help='数据文件名')
    arg.add_argument('--save_dir',default= "./ContrastiveLearning/Result/",help='结果存储路径')
    arg.add_argument('--date', default=[(0,20),(21,31)],
                    help='[(tra_date1,tra_enddate),(val_date1,val_enddate)]')
    arg.add_argument('--date_ClosedSet',default=[(0,10),(11,20),(21,31)])
    arg.add_argument('--period',default=3600*24,help='period s的行为记录构成一个样本')
    arg.add_argument('--item_size', default=160,help='词典大小')
    arg.add_argument('--emb_len',default=300,help='一个词要变成多少维度的')
    arg.add_argument('--max_seq_length',default=300,help='输入网络的序列的长度最大值')
    arg.add_argument('--users', default=100, help='待识别用户数')
    arg.add_argument('--margin', default=0.1)
    ## 网络设置
    arg.add_argument('--num_kernels',default=256,help='CNN的Channel output,初始channel是emb_len')
    arg.add_argument('--kernel_size',default=3,help='卷积核的大小，模型里面有padding 只要不是余2为1就可以')
    arg.add_argument('--pooling_stride',default=2,help='金字塔结构 池化层的步数')
    arg.add_argument('--blocks',default=4,help='金字塔结构(Max pool) 多少层(6)')
    arg.add_argument('--dp',default=0.5,help='dropout')
    arg.add_argument('--representation',default=300,help='最后emb网络输出的维度,300')
    arg.add_argument("--initializer_range",type=float,default=0.02)
    # 训练参数
    # train args
    arg.add_argument("--lr", type=float, default=0.00025, help="learning rate of adam")
    arg.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    arg.add_argument("--B_samples", type=int, default=5, help="每个类别抽多少样本")
    arg.add_argument("--B_classes", type=int, default=100, help="每个Batch抽多少类别")
    arg.add_argument("--epochs", type=int, default=200, help="number of epochs")
    arg.add_argument("--cuda_condition",default=True)
    # arg.add_argument("--no_cuda", action="store_true")
    # arg.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    arg.add_argument("--seed", default=19990605, type=int)
    arg.add_argument("--weight_decay", type=float, default=0.12, help="weight_decay of adam")
    arg.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    arg.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    # arg.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # 参数解析
    cfg = arg.parse_args('') #Jupyter Notebook的格式
    # cfg = arg.parse_args()  # Pycharm的格式
    return cfg