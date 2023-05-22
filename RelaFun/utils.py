import torch
import torch.nn as nn
from tqdm import tqdm
import random
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        # self.projector = nn.Linear(emb_len,emb_len)

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

def item_mask(item_seq,mask_ratio,mask_item_token):
    aug_seq = item_seq.clone()
    bsz = item_seq.size(0)
    vlen = (item_seq > 0).long().sum(dim=1)
    for i in range(bsz):
      for j in range(vlen[i]):
        prob = random.random()
        if prob < mask_ratio:
          aug_seq[i,j] = mask_item_token
    return aug_seq

def train_epoch(train_loader,model,optim,loss_fn,epoch,epochs,mask_token = None,mask_ratio=None,Acc=False):
    model.train()
    total_loss = 0.0
    total_batch = 0.0
    total_acc = 0.0
    total_sample = 0.0
    tra_bar = tqdm(train_loader)
    for inputs,hist_inputs,target in tra_bar:
        # inputs = anchor,positive,negtive
        bsz = target.shape[0]
        if mask_ratio is not None:
            # 只用一个mask
            # inputs_aug = item_mask(inputs, mask_ratio, mask_token)
            # inputs_aug = inputs_aug.to(device)
            # 使用多个mask
            inputs_aug_1 = item_mask(inputs,mask_ratio,mask_token)
            inputs_aug_2 = item_mask(inputs,mask_ratio,mask_token)
            inputs_aug = torch.cat([inputs_aug_1, inputs_aug_2], dim=0)
            # target = target.repeat(2) # 多个mask,标签也需要复制
        else:
            # inputs_aug = inputs.to(device)
            pass


        model.zero_grad()
        if target==[]:
            outputs = model.forward(*inputs)
            error = loss_fn(*outputs)
        else:
            outputs = model.forward(inputs_aug)
            f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
            outputs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            # # 1. 不考虑distance--代码
            # dis = model.distance(hist_inputs)
            # error = loss_fn(outputs, target)
            # # 2. 不考虑distance--dis 置为0
            # dis = 0
            # error = loss_fn(outputs, dis, target)
            # # 2. 考虑从distance（用固定的hist）
            # dis = model.distance(hist_inputs)
            # error = loss_fn(outputs, dis, target)
            # 3. 考虑distance，每个batch 有mask ,每个batch 重新统计得到的结果
            dis = model.distance(inputs_aug)
            f1, f2 = torch.split(dis, [bsz, bsz], dim=0)
            dis = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            error = loss_fn(outputs,dis,target)
            # 4.
            # error = loss_fn(outputs,target)

        error.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 防止梯度爆炸的
        optim.step()
        total_loss += error.item()
        total_batch += 1
        if Acc:
            total_acc += (outputs.argmax(dim=1) == target).sum().item()
            total_sample += target.shape[0]
            tra_bar.set_description('Train Epoch:[{}/{}], lr:{:.6f}, Loss:{:.4f}, Acc:{:.4f}'
                                    .format(epoch, epochs, optim.param_groups[0]['lr'], total_loss / total_batch,
                                            total_acc / total_sample))
        else:
            tra_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, epochs, optim.param_groups[0]['lr'], total_loss/total_batch ))

    return model,total_loss

def val_epoch(val_loader, model, optim, loss_fn, epoch, epochs,mask_token = None,mask_ratio=None,Acc=False):
    model.eval()
    total_loss = 0.0
    total_batch = 0.0
    total_acc = 0.0
    total_sample = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for inputs,hist_inputs,target in val_bar:
            # inputs = anchor,positive,negtive
            bsz = target.shape[0]
            # 只用一个mask
            # inputs_aug = item_mask(inputs, mask_ratio, mask_token)
            # inputs_aug = inputs_aug.to(device)
            # 使用多个mask
            inputs_aug_1 = item_mask(inputs, mask_ratio, mask_token)
            inputs_aug_2 = item_mask(inputs, mask_ratio, mask_token)
            inputs_aug = torch.cat([inputs_aug_1, inputs_aug_2], dim=0)
            if target==[]:
                outputs = model.forward(*inputs)
                error = loss_fn(*outputs)
            else:

                outputs = model.forward(inputs_aug)
                f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
                outputs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                # # 1. 不考虑distance--代码
                # dis = model.distance(hist_inputs)
                # error = loss_fn(outputs, target)
                # # 2. 不考虑distance--dis 置为0
                # dis = 0
                # error = loss_fn(outputs, dis, target)
                # # 2. 考虑从distance（用固定的hist）
                # dis = model.distance(hist_inputs)
                # error = loss_fn(outputs, dis, target)
                # 3. 考虑distance，每个batch 有mask ,每个batch 重新统计得到的结果
                dis = model.distance(inputs_aug)
                f1, f2 = torch.split(dis, [bsz, bsz], dim=0)
                dis = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                error = loss_fn(outputs, dis, target)
                # 4.
                # error = loss_fn(outputs,target)
            total_loss += error.item()
            total_batch += 1
            if Acc:
                total_acc += (outputs.argmax(dim=1) == target).sum().item()
                total_sample += target.shape[0]
                val_bar.set_description('Val Epoch:[{}/{}], lr:{:.6f}, Loss:{:.4f}, Acc:{:.4f}'
                                        .format(epoch, epochs, optim.param_groups[0]['lr'], total_loss / total_batch,
                                                total_acc / total_sample))
            else:
                val_bar.set_description(
                    'Val Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, epochs, optim.param_groups[0]['lr'],
                                                                        total_loss / total_batch))

    return model, total_loss
# test using a knn monitor
def test_epoch_knn(net, memory_data_loader, test_data_loader, epoch, epochs,knn_k,knn_t):
    net.eval()
    classes = max(memory_data_loader.dataset.labels)+1
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data,hist_data,target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net.get_representation(data)
            # feature = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.labels)

        # 对feature_bank 求 centroid
        feature_bank_centroid = []
        labels_centroid = []
        labels = set(memory_data_loader.dataset.labels)
        for label in labels:
            indices = torch.where(feature_labels==label,True,False)
            feature_bank_centroid.append(torch.mean(feature_bank[:,indices],dim=1).tolist())
            labels_centroid.append(label)
        feature_bank_centroid = torch.tensor(feature_bank_centroid).t()
        labels_centroid = torch.tensor(labels_centroid)

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data,hist_data,target in test_bar:
            # data, target = data, target
            feature = net.get_representation(data)
            # feature = net(data)
            feature = F.normalize(feature, dim=1)

            # pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t)
            pred_labels = knn_predict(feature, feature_bank_centroid, labels_centroid, classes, knn_k, knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # compute KL similarity between each feature vector and feature bank ----> [B,N],不能这样算 因为输出不保证为正数
    # from scipy.spatial.distance import cdist
    # import numpy as np
    # sim_matrix = 2-np.square(cdist(feature.numpy(),feature_bank.t().numpy(),"jensenshannon"))*2
    # sim_matrix = torch.from_numpy(sim_matrix)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # sim_labels = torch.tensor(sim_labels,dtype=torch.long)
    sim_labels = sim_labels.long()
    sim_weight = (sim_weight / knn_t).exp()


    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # one-hot-label [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def lossplot(loss_tra,loss_val):
    linemarker1 = [':^', ':X']
    linemarker2 = ['-s', '-o']
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    # ymajorLocator = MultipleLocator(0.1)  # 将主刻度设置为1的倍数
    # yminorLocator = MultipleLocator(0.02)  # 将次刻度设置为0.1的倍数
    # xmajorLocator = MultipleLocator(100)  # 将主刻度设置为100的倍数
    # xminorLocator = MultipleLocator(50)  # 将次刻度设置为50的倍数

    J = 0
    plt.figure()
    h = plt.subplot(1, 1, 1)
    # h.yaxis.set_major_locator(ymajorLocator)
    # h.yaxis.set_minor_locator(yminorLocator)
    # h.xaxis.set_major_locator(xmajorLocator)
    # h.xaxis.set_minor_locator(xminorLocator)
    h.xaxis.grid(True, which='major')
    h.yaxis.grid(True, which='major')

    epochs = len(loss_tra)
    x = range(epochs)
    plt.plot(x, loss_tra, linemarker2[0], linewidth=2, markersize=8, label='tra_loss')
    plt.plot(x, loss_val, linemarker2[1], linewidth=2, markersize=8, label='val_loss')

    plt.xlabel('Epochs')
    plt.ylabel('loss')
    # plt.ylim((y_min, y_max))
    plt.legend(loc=3)
    plt.tight_layout()
    plt.show()

def plot_embedding_2D(data, label, title):
    # tSNE降维结果
    from sklearn.manifold import TSNE
    import numpy as np
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    data = tsne_2D.fit_transform(data)
    # 绘制图形
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9}) #cm = color map
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return fig