#
# from torch.nn import Transformer
# model = Transformer()

import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from RelaFun.modules import TransformerEncoder,LayerNorm
class LSTMEmbedModel(nn.Module):
    def __init__(self, args):
        super(LSTMEmbedModel,self).__init__()
        """
        GRU 使用详解：https://blog.csdn.net/leitouguan8655/article/details/120219120
        LSTM 使用详解：https://blog.csdn.net/mch2869253130/article/details/103312364/
        Keras的GRU 使用详解：https://www.jianshu.com/p/15f2447fbc0f
        """
        self.item_size = args.item_size  # int 词典大小
        self.emb_len = args.emb_len #int 一个词要变成多少维度的
        self.hidden_size = args.hidden_size  # LSTM网络输出的维度
        self.representation = args.representation  # 最终向量表征的长度
        self.dp_prob = args.dp_prob
        self.lstm_layers = args.LSTM_layers
        self.bidirectional = True

        self.emb = nn.Embedding(self.item_size, self.emb_len, padding_idx=0)
        self.seq = nn.LSTM(input_size=self.emb_len, hidden_size=self.hidden_size,
                           num_layers=self.lstm_layers,batch_first=True,bidirectional=self.bidirectional) #[batch_size,seq_len,embedding_size]
        self.dropout = nn.Dropout(self.dp_prob)
        if self.bidirectional:
            self.Dense = nn.Linear(self.hidden_size*2,self.representation)
        else:
            self.Dense = nn.Linear(self.hidden_size, self.representation)

    def forward(self, item_seq):
        # item_seq = [B,seq_len]
        item_seq_emb = self.emb(item_seq) #返回维度[B,seq_len,emb_len]
        item_seq_hidden,_ = self.seq(item_seq_emb) #返回维度为[B,seq_len,emb_len]
        item_seq_hidden = torch.mean(item_seq_hidden,dim=1) # 返回维度为[B,emb_len]
        item_seq_repre = self.Dense(self.dropout(item_seq_hidden)) #返回维度为[B,representation]
        # 最终结果2范数和为1
        item_seq_repre = nn.functional.normalize(item_seq_repre,dim=1)
        return item_seq_repre

    def get_representation(self, item_seq):
        return self.forward(item_seq)
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            # module.weight.data.uniform_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class TransformerEmbedModel(nn.Module):
    def __init__(self,args):
        super(TransformerEmbedModel, self).__init__()
        self.item_size = args.item_size #int 词典大小
        self.emb_len= args.emb_len #int 一个词要变成多少维度的
        self.max_seq_length = args.max_seq_length # int 序列长度
        self.num_atten_heads = args.num_atten_heads # 多头注意力机制有多少头
        self.atten_dp_prob = args.atten_dp_prob # self_atten后面也接了一个dropout
        self.dp_prob = args.dp_prob # feedforward 部分的dropout
        self.hidden_act = args.hidden_act # feedforward 中全连接网络的激活函数
        self.hidden_size = args.hidden_size # feedforward 中全连接网络的中间层大小

        self.item_embeddings = nn.Embedding(self.item_size, self.emb_len, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_seq_length, self.emb_len, padding_idx=0)

        self.encoder = TransformerEncoder(args)
        self.LayerNorm = LayerNorm(self.emb_len, eps=1e-12) #对最后一个维度求均值 方差
        self.dropout = nn.Dropout(self.dp_prob)
        self.args = args

        self.cp_norm = nn.Linear(self.emb_len,self.emb_len)
        self.apply(self.init_weights)

        self.BN = nn.BatchNorm1d(self.emb_len)
        self.projector = nn.Linear(self.emb_len,256)

    # fine-tune same as SASRec
    def get_representation(self, item_seq):
        # 位置_mask
        attention_mask = (item_seq > 0).long()  # .long 把True变成1，item_seq>0 就是没padding的部分

        # position_embedding
        seq_length = item_seq.size(1) #item_seq = [B,seq_len]
        position_ids = torch.arange(start=0,end=seq_length,
                                    dtype=torch.long, device=item_seq.device) #[seq_length]
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq) #[2B,seq_len] 每一行都是[0,1,2,...,seq_length-1]
        # # position 也加上遮挡，注意position_embeddings这个地方得是seq_len+1
        # position_ids = torch.arange(start=1, end=seq_length + 1,
        #                             dtype=torch.long, device=item_seq.device)  # [seq_length]
        # position_ids = position_ids.unsqueeze(0).expand_as(item_seq)  # [2B,seq_len] 每一行都是[0,1,2,...,seq_length-1]
        #
        # position_ids = position_ids.mul(attention_mask)
        position_emb = self.position_embeddings(position_ids)  # [B L H] H是Embedding 向量的长度

        # item embedding
        item_seq_emb = self.item_embeddings(item_seq)
        item_seq_emb = item_seq_emb + position_emb
        item_seq_emb = self.LayerNorm(item_seq_emb)  # W * 标准化(item_seq_emb) + b
        item_seq_emb = self.dropout(item_seq_emb)


        # item sequence attention

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64 # [B 1 1 Len]
        max_len = attention_mask.size(-1) #和seq_length有什么不一样？
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # [1,L,L] torch.uint8 ,返回2-D张量的上三角部分,即上三角部分为1，其余地方为0
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)  # [1 1 len len] 下三角和对角线为0的部分
        subsequent_mask = subsequent_mask.long()
        # attention_mask
        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask #[B,1,1,Len] *[1,1,len,len] = [B,1,Len,Len] [1,Len]*[Len,Len]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 #[B,1,L,L]
        #extended_attention_mask 不仅是第一行是第一列为0，其它地方是-10000，第二行是前两列为0，其他列是-10000；
        # 超过padding部分的值也为-10000

        #
        sequence_output = self.encoder(item_seq_emb,
                                            extended_attention_mask,
                                            output_all_encoded_layers=False)[0] # [B,E]
        # output_all_encoded_layers=False的时候只有最后一层，所以[0]就是结果
        return sequence_output

    def distance(self,batch):
      # unique, counts = torch.unique(batch, return_counts=True, dim=1)
      B, seq_len = batch.shape
      # device = batch.device
      # hist_batch = torch.zeros((B,self.item_size), dtype=torch.int,device=device)
      # for i in range(B):
      #   sample = batch[i, :]
      #   for item_idx in sample:
      #     hist_batch[i, item_idx] += 1
      #      B,seq_len = batch.shape
      device = batch.device
      hist_batch = torch.zeros((B,self.item_size), dtype=torch.long,device=device) #tensor的idx必须是long类型
      for i in range(B):
          sample = batch[i,:]
          item_idx, counts = torch.unique(sample,return_counts=True)
          hist_batch[i,item_idx] = counts
      # dis = cdist(hist_batch.cpu(),hist_batch.cpu(),'jensenshannon')
      # dis = np.square(cdist(hist_batch[:,1:].cpu(), hist_batch[:,1:].cpu(), 'jensenshannon')) # 不用统计 padding
      # dis = np.square(cdist(hist_batch[:,1:-1].cpu(), hist_batch[:,1:-1].cpu(), 'jensenshannon')) # 不用统计 padding 和 mask_token
      # dis = cdist(hist_batch[:,1:-1].cpu(), hist_batch[:,1:-1].cpu(), 'cosine')
      dis = cdist(hist_batch[:,1:].cpu(), hist_batch[:,1:].cpu(), 'jensenshannon')
      return torch.tensor(dis,dtype=torch.float,device=device)
    def forward(self, item_seq):
        seq_representation = self.get_representation(item_seq)
        # seq_representation = self.BN(seq_representation)
        # seq_representation = self.projector(seq_representation)
        self_representation = nn.functional.normalize(seq_representation,dim=1)
        return seq_representation

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            # module.weight.data.uniform_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class DPCNNEmbedModel(nn.Module):
    """
    Reference:
        Deep Pyramid Convolutional Neural Networks for Text Categorization
    """

    def __init__(self, args):
        super(DPCNNEmbedModel, self).__init__()
        self.item_size = args.item_size
        self.emb_len = args.emb_len
        self.num_kernels = args.num_kernels  # 本来的Channel是emb_len, num_kernels就是输出的channel
        self.kernel_size = args.kernel_size  # 一般是3 ， 5 便于计算
        self.pooling_stride = args.pooling_stride
        self.radius = int(self.kernel_size / 2)
        assert self.kernel_size % 2 == 1, "DPCNN kernel should be odd!"
        self.blocks = args.blocks  # 金字塔结构有多少层
        self.dp = args.dp  # dropout的概率
        self.representation = args.representation
        self.apply(self.init_weights)
        self.token_embedding = nn.Embedding(self.item_size, self.emb_len, padding_idx=0)

        self.convert_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.emb_len, self.num_kernels,
                self.kernel_size, padding=self.radius)
        )
        self.convs = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius)
        ) for _ in range(self.blocks + 1)])

        # self.dropout = nn.Dropout(self.dp)

        # self.projector = nn.Sequential(
        #     torch.nn.Linear(self.num_kernels,self.num_kernels),
        #     # torch.nn.Dropout(self.dp),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.num_kernels,self.representation)
        # )

        self.linear = torch.nn.Linear(self.num_kernels, self.representation)

    def get_representation(self, batch):
        # batch = [B,L]
        embedding = self.token_embedding(batch)  # [B,L,E]
        embedding = embedding.permute(0, 2, 1)  # [B,E,L]
        conv_embedding = self.convert_conv(embedding)  # [B,num_kernels,L]
        conv_features = self.convs[0](conv_embedding)  # [B,num_kernels,L]
        conv_features = conv_embedding + conv_features  # [B,num_kernels,L]
        for i in range(1, len(self.convs)):  # F.max_pool1d
            block_features = F.max_pool1d(
                conv_features, self.kernel_size, self.pooling_stride)
            conv_features = self.convs[i](block_features)
            conv_features = conv_features + block_features  # [B,num_kernels,L']
        doc_embedding = F.max_pool1d(
            conv_features, conv_features.size(2)).squeeze()  # L'里面选一个最大值代表最终结果=[B,num_kernels,1],squeeze将1消除
        doc_embedding = self.linear(doc_embedding)
        # doc_embedding = self.linear(self.dropout(doc_embedding)) #每一个kernel的结果都很重要，不需要dropout.若有：78%--->50%
        return doc_embedding

    def distance(self, batch):
        # batch=[B,seq_len]
        B, seq_len = batch.shape
        device = batch.device
        hist_batch = torch.zeros((B, self.item_size), dtype=torch.long, device=device)  # tensor的idx必须是long类型
        for i in range(B):
            sample = batch[i, :]
            item_idx, counts = torch.unique(sample, return_counts=True)
            hist_batch[i, item_idx] = counts
        # dis = cdist(hist_batch.cpu(),hist_batch.cpu(),'jensenshannon')
        # dis = cdist(hist_batch[:,1:].cpu(), hist_batch[:,1:].cpu(), 'jensenshannon') # 不用统计 padding
        # dis = cdist(hist_batch[:,1:-1].cpu(), hist_batch[:,1:-1].cpu(), 'jensenshannon') # 不用统计 padding 和 mask_token; tes-ourloss+mask0.1=80% 差了，就是说mask是需要统计的
        # dis = cdist(hist_batch[:,1:-1].cpu(), hist_batch[:,1:-1].cpu(), 'cosine')
        # dis = np.square(cdist(hist_batch[:,1:-1].cpu(), hist_batch[:,1:-1].cpu(), 'jensenshannon'))*4 #欧式距离的范围为0~4，效果很差，还不如不加
        dis = cdist(hist_batch[:, 1:-1].cpu(), hist_batch[:, 1:-1].cpu(), 'jensenshannon')
        # dis = cdist(hist_batch[:,1:-1].cpu(), hist_batch[:,1:-1].cpu(), 'cosine')
        np.nan_to_num(dis, nan=0.0, posinf=0, neginf=0, copy=False)  # 对于一些可能算不出来的值，设置为0
        return torch.tensor(dis, dtype=torch.float, device=device)

    def forward(self, batch):
        # batch = [B,L]
        doc_embedding = self.get_representation(batch)
        # # doc_embedding = self.dropout(doc_embedding)
        # doc_embedding = self.projector(doc_embedding)
        doc_embedding = F.normalize(doc_embedding, dim=1)
        return doc_embedding

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            # module.weight.data.uniform_()
            # nn.init.xavier_uniform_(module.weight)
            # nn.init.kaiming_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
