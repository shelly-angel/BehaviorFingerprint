import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class CNNClassifier(nn.Module):
    def __init__(self, inputsize,Classes,num_kernels=16,pooling_stride = 2,kernel_size = 3):
        super(CNNClassifier, self).__init__()
        self.num_kernels = num_kernels
        self.pooling_stride = pooling_stride
        self.kernel_size = kernel_size
        self.radius = int(self.kernel_size / 2)
        assert self.kernel_size % 2 == 1, "kernel should be odd!"
        self.convert_conv = torch.nn.Conv1d(1, self.num_kernels,
                self.kernel_size, padding=self.radius)
        self.convs = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Conv1d(self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius),
            torch.nn.GELU(),
            torch.nn.Conv1d(self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius))
        self.dropout = nn.Dropout(0.9)
        self.output_layer = torch.nn.Linear(self.num_kernels*inputsize, Classes)

    def forward(self, embedding):
        m = embedding.size()[0]
        embedding = (embedding-embedding.mean(dim=1).reshape(m,-1))/(embedding.std(dim=1).reshape(m,-1))
        embedding = torch.unsqueeze(embedding, dim=1)
        conv_embedding = self.convert_conv(embedding)
        conv_features = self.convs(conv_embedding)
        conv_features = conv_embedding + conv_features
        out = conv_features.view(conv_features.size()[0],-1) #flatten
        outputs = self.output_layer(self.dropout(out))
        return outputs

class MLPClassifier(nn.Module):
    def __init__(self,NeuralNum):
        """
        :param NeuralNum: [featureNum,Hidden1,Hidden2,....,Classes]
        """
        super(MLPClassifier,self).__init__()
        N = len(NeuralNum)-1-1
        model=[]
        for i in range(N):
            model.append(nn.Linear(NeuralNum[i],NeuralNum[i+1]))
            model.append(nn.BatchNorm1d(NeuralNum[i+1]))
            model.append(nn.GELU())
            model.append(nn.Dropout(0.5))
        model.append(nn.Linear(NeuralNum[N],NeuralNum[N+1]))
        self.model = nn.ModuleList(model)

    def forward(self,input):
        m = input.size()[0]
        input = (input-input.mean(dim=1).reshape(m,-1))/(input.std(dim=1).reshape(m,-1))
        for i,l in enumerate(self.model):
            input = l(input)
        return input