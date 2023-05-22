from itertools import combinations
import torch.nn.functional as F
import numpy as np
import torch


def pdist(vectors):
    # distance_matrix[i,j]=-2*vectors[i]*vectors[j]+||vectors[i]||+||vectors[j]|| 要求范数为1的Euc才时这样计算 Euc(x,y) = sqrt(2-Cos(x,y))
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix
def pdist_Euc(vectors):
    distance_matrix = (vectors.unsqueeze(1)-vectors.t().unsqueeze(0)).pow(2).sum(dim=2)
    return distance_matrix
def pdist_Cos(vectors):
    distance_matrix = 1-vectors.mm(torch.t(vectors))
    return distance_matrix

class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


# class AllPositivePairSelector(PairSelector):
#     """
#     原始下载的样子，其中get_pairs 实际上没有用到embedding，所以修改版将这个去掉了
#     Discards embeddings and generates all possible pairs given labels.
#     If balance is True, negative pairs are a random sample to match the number of positive samples
#     """
#     def __init__(self, balance=True):
#         super(AllPositivePairSelector, self).__init__()
#         self.balance = balance
#
#     def get_pairs(self, embeddings, labels):
#         labels = labels.cpu().data.numpy()
#         all_pairs = np.array(list(combinations(range(len(labels)), 2)))
#         all_pairs = torch.LongTensor(all_pairs)
#         positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
#         negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
#         if self.balance:
#             negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]
#
#         return positive_pairs, negative_pairs

class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs

def BalancePositivePairSelector(): return AllPositivePairSelector(balance=True)

class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


# def hardest_negative(loss_values):
#     # ap_loss-an_loss+margin = loass_values, 选最难判别出来的样本对 anchor,positive,negtive index；
#     hard_negative = np.argmax(loss_values)
#     return hard_negative if loss_values[hard_negative] > 0 else None
def hardest_negative(loss_values):
    # ap_loss-an_loss+margin = loass_values, 选最难判别出来的样本对 anchor,positive,negtive index；
    hard_negative = torch.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0] #选负样本的距离不够远的（没有超过正样本）；anchor 和 负样本很近的
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None
def semihard_negative(loss_values, margin):
    # 选正样本距离-负样本距离在 margin内的，就是说正负样本差距不够
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

def tripletloss(ap_distance,an_distance,margin,fill1,fill2,fill3):
    return ap_distance - an_distance + margin
# def tri_oursloss_m1(ap_distance,an_distance,margin,anchor_positive,negative_indices,dis):
#     hist_distance = F.relu(dis[anchor_positive[0],negative_indices]-dis[anchor_positive[0],anchor_positive[1]])
#     loss_values = ap_distance - an_distance + margin + hist_distance
#     return loss_values
def tri_oursloss(ap_distance,an_distance,margin,anchor_positive,negative_indices,dis):
    # 2.
    loss_ap = F.relu(ap_distance-dis[anchor_positive[0],anchor_positive[1]])
    loss_an = F.relu(dis[anchor_positive[0],negative_indices]-an_distance)
    loss_values = ap_distance-an_distance+margin+loss_an+loss_ap
    return loss_values

def tri_ourloss_m2(ap_distance,an_distance,margin,anchor_positive,negative_indices,dis):
    tmp = F.relu(dis[anchor_positive[0], negative_indices] - dis[anchor_positive[0], anchor_positive[1]]-margin)+margin
    return ap_distance-an_distance+tmp

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin,negative_selection_fn,loss_value_fn,device="cpu"):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.device = device
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.loss_value_fn = loss_value_fn
        # self.disfun = disfun

    def get_triplets(self,embeddings,dis,labels):
        device = embeddings.device
        embeddings = embeddings
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0] #是label类的index
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0] #不是label类的index
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = torch.tensor(anchor_positives,dtype=torch.long,device=device)
            negative_indices = torch.tensor(negative_indices,dtype=torch.long,device=device)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]] #提取anchor 和 positive的距离
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                an_distance = distance_matrix[anchor_positive[0], negative_indices]
                loss_values = self.loss_value_fn(ap_distance,an_distance,self.margin,anchor_positive,negative_indices,dis) #loss_values=[1,len(neg)]
                hard_negative = self.negative_selection_fn(loss_values) #返回ap_dis-an_dis最大的index
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])


        triplets = torch.tensor(triplets,dtype=torch.long,device=device)
        return triplets #这里返回的都是row-index,batch的embedding的size=[B,E]


def HardestNegativeTripletSelector(margin, device="cpu"): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 loss_value_fn=tripletloss,device=device)
# def HardestNegativeTripletSelector_ourloss1(margin, device="cpu"): return FunctionNegativeTripletSelector(margin=margin,
#                                                                                  negative_selection_fn=hardest_negative,
#                                                                                  loss_value_fn=tri_oursloss_m1,device=device)
def HardestNegativeTripletSelector_ourloss(margin, device="cpu"): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 loss_value_fn=tri_oursloss,device=device)
def HardestNegativeTripletSelector_ourloss_m2(margin, device="cpu"): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 loss_value_fn=tri_ourloss_m2,device=device)


def RandomNegativeTripletSelector(margin, device="cpu"): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                device=device)


def SemihardNegativeTripletSelector(margin, device="cpu"): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  device=device)
