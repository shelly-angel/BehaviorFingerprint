# 连接云盘和GPU
from google.colab import drive
drive.mount('/content/gdrive') ##挂载谷歌云盘
import os
os.chdir('/content/gdrive/MyDrive/Colab_Notebooks')	##设置工作路径
import sys
sys.path.append('/content/gdrive/MyDrive/Colab_Notebooks/ContrastiveLearning')
print(os.path.abspath('.')) #输出绝对路径
print(os.listdir(path='.')) #当前路径下的文件夹有哪些
# 设置GPU
import torch
gpu_info = !nvidia-smi -i 0
gpu_info = '\n'.join(gpu_info)
print(gpu_info)
if torch.cuda.is_available(): # true 查看GPU是否可用
  print(torch.cuda.device_count()) #GPU数量， 1
  print(torch.cuda.current_device()) #当前GPU的索引， 0
  print(torch.cuda.get_device_name(0)) #输出GPU名称
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据输入
import pandas as pd
import numpy as np
import  time
from RelaFun.setRandomSeed import set_random_seed
from RelaFun.SequenceData import DataProcessing
from torch.utils.data import DataLoader
from RelaFun.datasets import behavSeq,BalancedBatchSampler,SupervisedSet
def GetData(cfg,device,random_choose=False):
  set_random_seed(seed=cfg.seed,deterministic=True)
  # 数据处理
  data = DataProcessing(cfg)
  user = data.Commonuser() #保证tra,val及test里面的用户是相同的
  print('用户数：',cfg.users)
  if random_choose:
    choose_user_idx = torch.randperm(len(user))[:cfg.users]
    choose_user = user[choose_user_idx]
  else:
    choose_user = user[:cfg.users]
  tra, val = data.dataset(choose_user)
  # cfg.item_size = max(set(tra.Item.values)|set(val.Item.values))+2 # 得到总共有多少种item,+1:padding, +2:padding, mask token
  tra = data.sequential(tra,period=cfg.period)
  val = data.sequential(val)
  # length_tra = [len(seq) for seq in tra.behav_type.values]
  # cfg.max_seq_length = min(max(length_tra),cfg.max_seq_length)
  # print('max_seq_length=,mean_length=,max_length=',cfg.max_seq_length,sum(length_tra)/len(length_tra),max(length_tra))
  tra_seq = behavSeq(tra,height=1,sequence_len=cfg.max_seq_length,day=False) # height = int(3600*24/cfg.period)
  tra_labels = np.array(tra_seq.id.values.tolist())
  train_batch_sampler = BalancedBatchSampler(tra_labels, n_classes=cfg.B_classes, n_samples=cfg.B_samples)
  tra_loader = DataLoader(SupervisedSet(tra,sequence_len=cfg.max_seq_length,
              items = cfg.item_size),batch_sampler=train_batch_sampler)
  memory_loader = DataLoader(SupervisedSet(tra,sequence_len=cfg.max_seq_length,
                items = cfg.item_size),batch_size=cfg.batch_size,shuffle=False)
  tes_loader = DataLoader(SupervisedSet(val,sequence_len=cfg.max_seq_length,
              items = cfg.item_size),batch_size=cfg.batch_size,shuffle=False)
  return tra_loader,memory_loader,tes_loader,cfg

#  网络训练及识别
import time
import numpy as np
from RelaFun.model import TransformerEmbedModel,DPCNNEmbedModel
import torch.optim as optim
from torch.optim import lr_scheduler
from RelaFun.EvaluationFuns import test_knn_Cos, test_knn_Euc
from RelaFun.utils import test_epoch_knn
def main(cfg, device,modelname='Trans'):
  # 数据处理
  tra_loader, memory_loader, tes_loader, cfg = GetData(cfg, device, random_choose=False)
  # 网络定义
  if modelname == 'Trans':
    embedding_net = TransformerEmbedModel(cfg)
  elif modelname == 'DPCNN':
    embedding_net = DPCNNEmbedModel(cfg)
  model = embedding_net.to(device)
  from RelaFun.selector import HardestNegativeTripletSelector, HardestNegativeTripletSelector_ourloss, \
    HardestNegativeTripletSelector_ourloss_m2
  from RelaFun.losses import OnlineTripletLoss, OnlineOurLoss, OnlineOurLoss_m2
  loss_fn = OnlineOurLoss_m2(cfg.margin, HardestNegativeTripletSelector_ourloss_m2(cfg.margin))
  optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
  scheduler = lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)
  # 网络训练
  from tqdm import tqdm
  def train_epoch(model, epoch, epochs):
    model.train()
    total_loss = 0.0
    total_batch = 0.0
    tra_bar = tqdm(tra_loader)
    for inputs, target in tra_bar:
      inputs, target = inputs.to(device), target.to(device)
      model.zero_grad()
      outputs = model.forward(inputs)
      dis = model.distance(inputs)
      error = loss_fn(outputs, dis, target)
      # error = loss_fn(outputs,torch.tensor([0]),target)
      error.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 防止梯度爆炸的
      optimizer.step()
      total_loss += error.item()
      total_batch += 1
      tra_bar.set_description('Train Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, epochs, total_loss))
    return model, total_loss

  loss_tra = []
  modelsavename = cfg.filename[:-4] + 'U' + str(cfg.users) + cfg.modelname
  import numpy as np
  checkpoint_interval = 5 # 防止意外退出，每隔5个epoch重新存储一下模型

  for e in range(cfg.epochs):
    model, loss_tra_tmp = train_epoch(model, e, cfg.epochs)
    loss_tra.append(loss_tra_tmp)
    scheduler.step()

    if (e + 1) % checkpoint_interval == 0: # 防止意外退出
      checkpoint = {"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": e}
      path_checkpoint = cfg.save_dir + 'models/' + modelsavename + "checkpoint.pkl"
      torch.save(checkpoint, path_checkpoint)

  res = test_knn_Cos(model, memory_loader, tes_loader, cfg.users, device)
  print(np.round(np.asarray(res), 4))

  torch.save(model.state_dict(), cfg.save_dir + 'models/' + modelsavename + '.pth')
  with open(cfg.save_dir + 'accs/' + cfg.filename[:-4] + cfg.modelname + '.txt', 'a+') as f:
    for num in res:
      f.write(str(round(num, 4)) + '\t')
    f.write('\n')
  f.close()
  np.savetxt(cfg.save_dir + 'loss/' + modelsavename + '.txt', np.array(loss_tra), fmt='%.4f')

  # res = test_knn_Euc(model,memory_loader,tes_loader,cfg.users,device)
  # print(np.round(np.asarray(res), 4))
  return list(np.round(np.asarray(res), 4))

if __name__ == '__main__':
  from RelaFun.NNConfig import Transconfig, DPCNNconfig
  cfg = Transconfig()
  cfg.filename = 'Reddit.txt'
  cfg.modelname = 'MyTrans'
  cfg.date = [(0, 20), (21, 31)] #tra,tes
  res = main(cfg, device, modelname='Trans')
  print(res)