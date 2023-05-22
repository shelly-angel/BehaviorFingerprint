import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances 欧式距离的平方
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

class OnlineTULSNLoss(nn.Module):
    """
    Online TULSN loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, alpha, pair_selector,margin=1):
        super(OnlineTULSNLoss, self).__init__()
        self.alpha = alpha #正则化参数
        self.pair_selector = pair_selector
        self.margin = margin

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(target) #实际上没有难例挖掘，只是让它们平衡
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        positive_tmp1 = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        positive_tmp2 = embeddings[positive_pairs[:, 0]].pow(2).sum(1)+embeddings[positive_pairs[:, 1]].pow(2).sum(1)
        positive_tmp3 = torch.abs(torch.abs(embeddings[positive_pairs[:,0]])-1).sum(1)+torch.abs(torch.abs(embeddings[positive_pairs[:,1]])-1).sum(1)
        positive_loss = 0.5 * (positive_tmp1/positive_tmp2)+self.alpha * (positive_tmp3/positive_tmp2)

        negative_tmp1 = (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(1)
        negative_tmp2 = embeddings[negative_pairs[:, 0]].pow(2).sum(1)+embeddings[negative_pairs[:, 1]].pow(2).sum(1)
        negative_tmp3 = torch.abs(torch.abs(embeddings[negative_pairs[:,0]])-1).sum(1)+torch.abs(torch.abs(embeddings[negative_pairs[:,1]])-1).sum(1)
        negative_loss = 0.5*(F.relu(self.margin-(negative_tmp1/negative_tmp2)))+self.alpha * (negative_tmp3/negative_tmp2)

        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings,dis,target):
        triplets = self.triplet_selector.get_triplets(embeddings,dis,target) #[B,3] 3:anchor,positive,negtive
        if embeddings.is_cuda:
            triplets = triplets.cuda()
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances +self.margin)
        return losses.mean()

class OnlineOurLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineOurLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings,dis,target):
        triplets = self.triplet_selector.get_triplets(embeddings,dis,target) #[B,3] 3:anchor,positive,negtive
        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        ap_hist_distances = dis[triplets[:, 0], triplets[:, 1]]
        an_hist_distances = dis[triplets[:, 0], triplets[:, 2]]
        losses = F.relu(ap_distances-an_distances+self.margin)\
                 +F.relu(ap_distances-ap_hist_distances)\
                 +F.relu(an_hist_distances-an_distances)
        return losses.mean()


class OnlineOurLoss_m2(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineOurLoss_m2, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings,dis,target):
        triplets = self.triplet_selector.get_triplets(embeddings,dis,target) #[B,3] 3:anchor,positive,negtive
        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        ap_hist_distances = dis[triplets[:, 0], triplets[:, 1]]
        an_hist_distances = dis[triplets[:, 0], triplets[:, 2]]
        # margin = ap_hist_distances-an_hist_distances
        margin = F.relu(an_hist_distances-ap_hist_distances-self.margin)+self.margin
        losses = F.relu(ap_distances-an_distances+margin)
        return losses.mean()

# class OnlineOurLoss_m1(nn.Module):
#     """
#     Online Triplets loss
#     Takes a batch of embeddings and corresponding labels.
#     Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
#     triplets
#     """
#     def __init__(self, margin, triplet_selector):
#         super(OnlineOurLoss_m1, self).__init__()
#         self.margin = margin
#         self.triplet_selector = triplet_selector
#     def forward(self, embeddings,dis,target):
#         triplets = self.triplet_selector.get_triplets(embeddings,dis,target) #[B,3] 3:anchor,positive,negtive
#         if embeddings.is_cuda:
#             triplets = triplets.cuda()
#         ap_hist_distances = dis[triplets[:,0],triplets[:,1]]
#         an_hist_distances = dis[triplets[:,0],triplets[:,2]]
#         hist_losses = F.relu(an_hist_distances-ap_hist_distances)
#         ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
#         an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
#         losses = F.relu(ap_distances - an_distances + hist_losses + self.margin)
#         return losses.mean()



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, feat_dim].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, feat_dim],'
                             'at least 2 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0],  -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # contrast_count = features.shape[1]
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # if self.contrast_mode == 'one':
        #     anchor_feature = features[:, 0]
        #     anchor_count = 1
        # elif self.contrast_mode == 'all':
        #     anchor_feature = contrast_feature
        #     anchor_count = contrast_count
        # else:
        #     raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # logits = torch.div(
        #     torch.matmul(features, features.T),
        #     self.temperature) #zyy -不要numerical stability 在IPTV 100的时候没有大的差别

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count) #anchor_count = contrast_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        ) #除去对角线全为1
        # torch.scatter(target, dim,index,src)，将一个源张量（src）中的值按照指定的轴方向（dim）和对应的位置关系（index）逐个填充到目标张量（target）
        negative_mask = 1-mask #zyy
        mask = mask * logits_mask #每一行 自身和不同类的地方为0 其余地方为1，相当于只有positive为正

        # compute log_prob
        exp_logits = torch.exp(logits)
        positive_logits = exp_logits * mask
        negative_logits = exp_logits * negative_mask #不完全是negative，只是不包含自己罢了
        # log_prob = logits - torch.log(negative_logits.sum(1, keepdim=True))
        log_prob = torch.log(positive_logits.sum(1)) - torch.log(negative_logits.sum(1))
        mean_log_prob_pos = log_prob / mask.sum(1)

        # # compute log_prob-zyy,差了一点 虽然差不到那里去
        # exp_logits = torch.exp(logits) * negtive_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # min_log_prob_pos = (mask * log_prob).min(dim=1).values

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(1, batch_size).mean()
        # loss = - (self.temperature / self.base_temperature) * min_log_prob_pos


        return loss.mean()

class SupConLoss_centroid(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_centroid, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, feat_dim].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, feat_dim],'
                             'at least 2 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0],  -1)

        batch_size = features.shape[0]
        feat_dim = features.shape[1]


        # compute centroid
        labels_idx = {}
        for idx in range(len(labels)):
            label = labels[idx].tolist()
            if label in labels_idx.keys():
                labels_idx[label].append(idx)
            else:
                labels_idx[label] = [idx]
        centroid = torch.zeros((len(labels_idx.keys()),feat_dim))
        labels_set = []
        i = 0
        for label in labels_idx.keys():
            labels_set.append(label)
            idx = labels_idx[label]
            # centroid.append(features[idx].mean(1).tolist())
            centroid[i,:] = features[idx].mean(0)
            i = i+1
        # centroid = torch.tensor(centroid) #[C,feat_dim]


        # compute sim
        anchor_dot_contrast = torch.div(
            torch.matmul(features, centroid.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            labels_set = torch.tensor(labels_set).contiguous().view(-1,1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            positive_mask = torch.eq(labels, labels_set.T).float().to(device)
            negative_mask = 1-positive_mask
        else:
            mask = mask.float().to(device)


        # compute log_prob
        positive_logits = logits*positive_mask
        negative_logits = torch.exp(logits) * negative_mask
        loss = torch.log(negative_logits.sum(1)) - positive_logits.sum(1)
        # loss = (self.temperature / self.base_temperature) * loss

        return loss.mean()

class SupConLoss_dis(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_dis, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, dis,labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, feat_dim].
            dis: [bsz,bsz]
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, feat_dim],'
                             'at least 2 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0],  -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            sameclass_mask = torch.eq(labels, labels.T).float().to(device)
            negative_mask = 1-sameclass_mask
        else:
            mask = mask.float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(sameclass_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )  # 除去对角线全为1
        # torch.scatter(target, dim,index,src)，将一个源张量（src）中的值按照指定的轴方向（dim）和对应的位置关系（index）逐个填充到目标张量（target）
        positive_mask = sameclass_mask * logits_mask  # 每一行 自身和不同类的地方为0 其余地方为1，相当于只有positive为正




        # compute logits
        sim = torch.matmul(features,features.T)
        sim_KL = 1-dis
        sim_gap = sim-sim_KL

        anchor_dot_contrast = torch.div(sim+sim_gap,self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # loss
        negative_logits = torch.exp(logits) * negative_mask
        log_prob = logits - torch.log(negative_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (sameclass_mask * log_prob).sum(1) / sameclass_mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # #
        # sim_KL = 1-dis
        # positive_loss = F.relu(positive_mask*(sim_KL-sim)).sum(1) #sim_KL应该比sim小，这样就全部是0
        # negative_loss = F.relu(negative_mask*(sim-sim_KL)).sum(1)

        # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))


        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        # min_log_prob_pos = (positive_mask * log_prob).min(dim=1).values

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos +positive_loss+negative_loss
        # loss = loss.view(1, batch_size).mean()
        # loss = - (self.temperature / self.base_temperature) * min_log_prob_pos
        # alpha,beta,gamma = 0.8,0.1,0.1
        # loss = -alpha*mean_log_prob_pos+beta*positive_loss+gamma*negative_loss


        return loss.mean()


import torch
import torch.nn as nn


class SupConLoss_Aug_dis(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_Aug_dis, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, dis,labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # mask = torch.eq(labels, labels.T).float().to(device)
            sameclass_mask = torch.eq(labels, labels.T).float().to(device)
            negative_mask = 1 - sameclass_mask
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_dis = torch.cat(torch.unbind(dis,dim=1),dim=0) #zyy
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            anchor_dis = contrast_dis #zyy
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        sim = torch.matmul(anchor_feature, contrast_feature.T)
        sim_KL = 1-contrast_dis
        sim_gap = sim-sim_KL
        anchor_dot_contrast = torch.div(sim+sim_gap,self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        sameclass_mask = sameclass_mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(sameclass_mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        positive_mask = sameclass_mask * logits_mask
        negative_mask = negative_mask.repeat(anchor_count,contrast_count)
        # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        exp_logits = torch.exp(logits) * negative_mask #添加了dis后 分母只能是negative,因为 分母要越小越好，分子要越大越好
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) #若需要将求和转换为乘积 可以用log(p1*p2*p3...pn)=logp1+logp2+logp3+...+logpn

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        # mean_log_prob_pos = (sameclass_mask * log_prob).sum(1) / sameclass_mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss_Aug(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_Aug, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # mask = torch.eq(labels, labels.T).float().to(device)
            sameclass_mask = torch.eq(labels, labels.T).float().to(device)
            negative_mask = 1 - sameclass_mask
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        sameclass_mask = sameclass_mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(sameclass_mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        positive_mask = sameclass_mask * logits_mask
        negative_mask = negative_mask.repeat(anchor_count,contrast_count)
        # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        exp_logits = torch.exp(logits) * negative_mask #添加了dis后 分母只能是negative,因为 分母要越小越好，分子要越大越好
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) #若需要将求和转换为乘积 可以用log(p1*p2*p3...pn)=logp1+logp2+logp3+...+logpn

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        mean_log_prob_pos = (sameclass_mask * log_prob).sum(1) / sameclass_mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
