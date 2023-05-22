import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler,SequentialSampler

def UnifyDataFormat(row,height,width):
    # 将数据规整到height*width的形式，如24(h) * 32(sequence_len),否则dataloader会报错
    res = np.zeros((height,width),dtype='int32')
    for i in range(len(row.t)):
        length =min(len(row.X[i]),width)
        res[row.t[i],:length] = row.X[i][:length]
    return res
def UnifyDataSeqlength(row,width):
    res = np.zeros((1,width),dtype='int32')
    length = min(len(row.behav_type),width)
    res[0,:length] = row.behav_type[:length]
    return res
def StatHist(row,fea_len):
    res = np.zeros(fea_len,dtype='int32')
    fea_seq = row.X[0] # 若只有一行
    for item in fea_seq:
        res[item] +=1
    return res[1:]
    # return res


# class TripletSet(Dataset):
#     def __init__(self,data,height,sequence_len):
#         behavSeq_grp = data.groupby(["id", "date"]).agg(
#             X =("behav_type", list),
#             t = ("time",list)
#         ).reset_index()
#         # 将维度设置为一样的
#         behavSeq_grp.X = behavSeq_grp.apply(UnifyDataFormat,
#                                             axis=1,args=(height,sequence_len))
#
#         self.X = np.array(behavSeq_grp.X.values.tolist())
#         self.labels = np.array(behavSeq_grp.id.values.tolist())
#         self.labels_set = set(self.labels)
#         self.label_to_indices = {label: np.where(self.labels == label)[0]
#                                  for label in self.labels_set}  # 每一类的index
#     def __len__(self):
#         return len(self.labels)
#     def __getitem__(self, index):
#         seq1,label1 = self.X[index],self.labels[index]
#         positive_index = index
#         while positive_index == index: #随机选择一个不是自己的正样本
#             positive_index = np.random.choice(self.label_to_indices[label1])
#         negative_label = np.random.choice(list(self.labels_set - set([label1])))
#         negative_index = np.random.choice(self.label_to_indices[negative_label])
#         seq2 = self.X[positive_index]
#         seq3 = self.X[negative_index]
#         return (
#             torch.tensor(seq1[0], dtype=torch.long),
#             torch.tensor(seq2[0], dtype=torch.long),
#             torch.tensor(seq3[0], dtype=torch.long)),[]

class TripletSet(Dataset):
    def __init__(self,data,height,sequence_len,items,day=False):
        if day:  # 若按天划分为矩阵型样本输入(height(如24h)*sequence_len)
            behavSeq_grp = data.groupby(["id", "date"]).agg(
                X=("behav_type", list),
                t=("time", list)
            ).reset_index()
            # 将维度设置为一样的
            behavSeq_grp.X = behavSeq_grp.apply(UnifyDataFormat,
                                                axis=1, args=(height, sequence_len))
            # 直方图统计
            self.hist = behavSeq_grp.apply(StatHist, axis=1, args=(items,)).values
            self.X = np.array(behavSeq_grp.X.values.tolist())
            self.labels = np.array(behavSeq_grp.id.values.tolist()) - 1
        else:
            data.X = data.apply(UnifyDataSeqlength, axis=1, args=(sequence_len,))
            self.hist = data.apply(StatHist, axis=1, args=(items,)).values
            self.X = np.array(data.X.values.tolist())
            self.labels = np.array(data.id.values.tolist()) - 1
            self.labels_set = set(self.labels)
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}  # 每一类的index
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        seq1,label1 = self.X[index],self.labels[index]
        positive_index = index
        while positive_index == index: #随机选择一个不是自己的正样本
            positive_index = np.random.choice(self.label_to_indices[label1])
        negative_label = np.random.choice(list(self.labels_set - set([label1])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        seq2 = self.X[positive_index]
        seq3 = self.X[negative_index]
        return (
            torch.tensor(seq1[0], dtype=torch.long),
            torch.tensor(seq2[0], dtype=torch.long),
            torch.tensor(seq3[0], dtype=torch.long))

def behavSeq(data,height,sequence_len,day=True):
    if day:
        # 保留每一天 time的时间
        behavSeq_grp = data.groupby(["id", "date"]).agg(
            X=("behav_type", list),
            t=("time", list)
        ).reset_index()
        # 将维度设置为一样的
        behavSeq_grp.X = behavSeq_grp.apply(UnifyDataFormat,
                                            axis=1, args=(height, sequence_len))
        return behavSeq_grp
    else:
        data['X'] = data.apply(UnifyDataSeqlength, axis=1, args=(sequence_len,))
        return data[['id','X']]

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}  # 每一类的index
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l]) #对label_to_indeices[l]的做shuffle操作，后续会改变
        self.used_label_indices_count = {label: 0 for label in self.labels_set} #对每个label下有多少样本做计数
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False) #在labels_set中无放回抽样n_classes个样本
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]): #下一次不够数了，从0开始
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices #它和return不一样，它是每一次执行生成一个 index,
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class SupervisedSet(Dataset):
    def __init__(self,data,sequence_len,items,need_date=False,need_stats=False):

        data.X = data.apply(UnifyDataSeqlength,axis=1,args=(sequence_len,))
        if need_stats:
            self.hist = data.apply(StatHist, axis=1, args=(items,)).values
        if need_date:
            self.date = np.array(data.date.values.tolist())
        self.X = np.array(data.X.values.tolist())
        self.labels = np.array(data.id.values.tolist())-1
        self.need_date = need_date
        self.need_stats = need_stats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq, label = self.X[index], self.labels[index]
        if self.need_stats:
            stats = self.hist[index]
            if self.need_date:
                d = self.date[index]
                return (torch.tensor(seq[0], dtype=torch.long), torch.tensor(stats, dtype=torch.long),
                                    torch.tensor(d, dtype=torch.long),torch.tensor(label, dtype=torch.long))
            else:
                return (torch.tensor(seq[0], dtype=torch.long), torch.tensor(stats, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long))
        if self.need_date:
            d = self.date[index]
            return (torch.tensor(seq[0], dtype=torch.long),torch.tensor(d, dtype=torch.long),torch.tensor(label, dtype=torch.long))
        else:
            return (torch.tensor(seq[0], dtype=torch.long),
                torch.tensor(label, dtype=torch.long))



class SupervisedSet2(Dataset):
    def __init__(self,data,height,sequence_len,items,need_date=False,need_stats=False):
        # data=[id,date,time,seq(list)]
        # 若按天划分
        behavSeq_grp = data.groupby(["id", "date"]).agg(
            X=("behav_type", list),
            t=("time", list)
        ).reset_index()
        # 将维度设置为一样的
        behavSeq_grp.X = behavSeq_grp.apply(UnifyDataFormat,
                                            axis=1, args=(height, sequence_len))
        # 直方图统计
        if need_stats:
            self.hist = behavSeq_grp.apply(StatHist, axis=1, args=(items,)).values
        self.X = np.array(behavSeq_grp.X.values.tolist())
        self.labels = np.array(behavSeq_grp.id.values.tolist()) - 1
        self.need_date = need_date
        self.need_stats = need_stats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq, label = self.X[index], self.labels[index] #seq 是矩阵形式的
        stats = self.hist[index]
        return (torch.tensor(seq, dtype=torch.long), torch.tensor(stats,dtype=torch.long),
            torch.tensor(label, dtype=torch.long))
from tqdm import tqdm
class BehaviorIdentify(Dataset):
    def __init__(self,data_loader,embedding_net):
        self.labels = torch.tensor(data_loader.dataset.labels,dtype=torch.long)
        self.net = embedding_net
        # 得到embedding的结果
        self.net.eval()
        self.X = []
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm(data_loader, desc='Feature extracting'):
                feature = self.net.get_representation(data)
                self.X.append(feature)
            self.X = torch.cat(self.X, dim=0).contiguous()
    def get(self):
        return self.X.numpy(),self.labels.numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return self.X[index,:],self.labels[index]

