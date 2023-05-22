import torch
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm

def FeatureBankToCentroid(feature_bank,feature_labels):
    feature_bank_centroid = []
    labels_centroid = []
    labels = set(feature_labels.tolist())
    for label in labels:
        indices = torch.where(feature_labels == label, True, False)
        feature_bank_centroid.append(torch.mean(feature_bank[:, indices], dim=1).tolist())
        labels_centroid.append(label)
    feature_bank_centroid = torch.tensor(feature_bank_centroid,device=feature_bank.device).t()
    labels_centroid = torch.tensor(labels_centroid,device=feature_bank.device)
    return feature_bank_centroid,labels_centroid
def metrics(acc_5,con_mat):
    label_counts = con_mat.sum(dim=1)  # 每一个类别的总样本数
    counts = label_counts.sum()  # 测试样本总数
    TP = con_mat.diagonal()  # 每一个类别 是这个类别且被识别为这个类别了
    FP = con_mat.sum(dim=0) - con_mat.diagonal()  # 每一个类别 不是这个类别的但是被识别为这个类别了
    TN = label_counts.sum() - label_counts - con_mat.sum(
        dim=0) + con_mat.diagonal()  # 每一个类别 不是这个类别的且没有被识别为这个类别 ;所有元素和-当前列和-当前行和
    FN = label_counts - con_mat.diagonal()  # 每一个类别 是这个类别的但是被识别为不是这个类别了


    acc_1 = TP.sum()/counts
    acc_5 = acc_5 / counts

    precision = torch.divide(TP, TP + FP)
    precision = torch.nan_to_num(precision, nan=0, posinf=0, neginf=0)
    recall = torch.divide(TP, TP + FN)
    recall = torch.nan_to_num(recall, nan=0, posinf=0, neginf=0)
    f1_score = torch.divide(2 * precision * recall, precision + recall)
    f1_score = torch.nan_to_num(f1_score, nan=0., posinf=0., neginf=0.)

    tmp = FP + TN
    FAR_micro = torch.divide(FP.sum(), tmp.sum())
    FAR_micro = torch.nan_to_num(FAR_micro, nan=0, posinf=0, neginf=0)
    FAR_macro = torch.divide(FP, tmp)
    FAR_macro = torch.nan_to_num(FAR_macro, nan=0, posinf=0, neginf=0)

    tmp = TP + FN
    FRR_micro = torch.divide(FN.sum(), tmp.sum())
    FRR_micro = torch.nan_to_num(FRR_micro, nan=0, posinf=0, neginf=0)
    FRR_macro = torch.divide(FN, tmp)
    FRR_macro = torch.nan_to_num(FRR_macro, nan=0, posinf=0, neginf=0)

    return [acc_1.tolist(), acc_5.tolist(), precision.mean().tolist(),
            recall.mean().tolist(), f1_score.mean().tolist(), FAR_micro.tolist(),
            FRR_micro.tolist(),FAR_macro.mean().tolist(), FRR_macro.mean().tolist()]
def evaluation(Sim,label_train,label_test,con_mat,top_k = 5,value=0.0,sim=True):
    # https://www.cvmart.net/community/detail/2840 micro 和 macro的介绍
    import warnings
    warnings.filterwarnings('ignore')
    Sim = torch.nan_to_num(Sim,nan = value,posinf=value,neginf=value) # 可能存在一些nan的值，全部替换为0
    idx = Sim.argmax(dim=1) if sim else Sim.argmin(dim=1)
    label_pred = label_train[idx]
    if label_test.device == 'cpu':
        con_mat_tmp = confusion_matrix(label_test, label_pred)  # 如果标签有3类(label_true里面最大的'数')，返回一个3*3的矩阵; 标签从小到大,返回的是一个array
    else:
        label_test_cpu,label_pred_cpu = label_test.cpu().numpy(),label_pred.cpu().numpy()
        con_mat_tmp = confusion_matrix(label_test_cpu, label_pred_cpu)
    con_mat_tmp = torch.tensor(con_mat_tmp,dtype=torch.int,device=label_test.device)
    labels = list(set(label_test.tolist()) | set(label_pred.tolist()))
    labels.sort() #从小到大排序，自身也改变了, 实际上就是idx
    I = 0
    for idx in labels:
        con_mat[idx,labels] += con_mat_tmp[I,:]
        I = I+1

    _,idx_k = Sim.topk(top_k,dim=1) #二维矩阵，第i行为test-i 用户在训练集用户里面前五个匹配得上的
    label_pred_k = label_train[idx_k]
    label_test_k = label_test.repeat((top_k,1)).t()
    acc_k = ((label_test_k==label_pred_k).sum(dim=1)>0).sum()

    return acc_k,con_mat
def test_knn_Cos(net,memory_data_loader,test_data_loader,classes,device="cpu"):
    net.eval()
    # 混淆矩阵初始化
    # con_mat = np.zeros((classes,classes),dtype='int32')
    acc5 = 0
    # 提取训练集的模板
    feature_bank = []
    with torch.no_grad():
        # generate feature bank
        for data,target in tqdm(memory_data_loader, desc='Feature extracting'):
            data,target = data.to(device),target.to(device)
            feature = net.get_representation(data)
            # feature = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.labels,device=feature_bank.device)
        # # [N]
        # feature_dates = torch.tensor(memory_data_loader.dataset.date,device=feature_bank.device)

        # centroid化
        feature_bank_centroid, labels_centroid = FeatureBankToCentroid(feature_bank, feature_labels)
        con_mat = torch.zeros((classes, classes), dtype=torch.int).to(feature_bank.device)
        # 提取测试集的模板
        test_bar = tqdm(test_data_loader)
        for data,target in test_bar:
            data, target = data.to(device), target.to(device)
            feature = net.get_representation(data)
            # feature = net(data)
            feature = F.normalize(feature, dim=1)

            # 计算相似度
            # sim_matrix = torch.mm(feature, feature_bank) #Cosine
            # acc1_tmp,acc5_tmp,con_mat_tmp = evaluation(sim_matrix, feature_labels, label_test=target,con_mat=con_mat,top_k=5, sim=True)

            sim_matrix = torch.mm(feature, feature_bank_centroid)  # Cosine
            acc5_tmp,con_mat_tmp = evaluation(sim_matrix, labels_centroid, label_test=target,con_mat=con_mat,top_k=5, sim=True)
            acc5 +=acc5_tmp
        # res = pd.DataFrame(metrics(acc1, acc5, con_mat),columns=['acc1','acc5','precision_macro','recall_macro',
        #                                                          'f1_score_macro','FAR_micro','FRR_micro','FAR_macro','FRR_macro'])
        res = metrics(acc5, con_mat)

    return res
def test_knn_Euc(net,memory_data_loader,test_data_loader,classes,device='cpu'):
    net.eval()
    # 混淆矩阵初始化
    # con_mat = np.zeros((classes,classes),dtype='int32')
    acc5 = 0
    # 提取训练集的模板
    feature_bank = []
    with torch.no_grad():
        # generate feature bank
        for data,target in tqdm(memory_data_loader, desc='Feature extracting'):
            data, target = data.to(device), target.to(device)
            feature = net.get_representation(data)
            # # feature = net(data)
            # feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.labels,device=feature_bank.device)
        # # [N]
        # feature_dates = torch.tensor(memory_data_loader.dataset.date,device=feature_bank.device)

        # centroid化
        feature_bank_centroid, labels_centroid = FeatureBankToCentroid(feature_bank, feature_labels)
        con_mat = torch.zeros((classes, classes), dtype=torch.int).to(feature_bank.device)
        # 提取测试集的模板
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature = net.get_representation(data)
            # # feature = net(data)
            # feature = F.normalize(feature, dim=1)

            # 计算相似度
            # sim_matrix = torch.mm(feature, feature_bank) #Cosine
            # acc1_tmp,acc5_tmp,con_mat_tmp = evaluation(sim_matrix, feature_labels, label_test=target,con_mat=con_mat,top_k=5, sim=True)
            sim_matrix = -1 * (feature.unsqueeze(1)-feature_bank_centroid.t().unsqueeze(0)).pow(2).sum(dim=2) #二范数
            acc5_tmp,con_mat_tmp = evaluation(sim_matrix, labels_centroid, label_test=target,con_mat=con_mat,top_k=5, sim=True)
            acc5 +=acc5_tmp
        # res = pd.DataFrame(metrics(acc1, acc5, con_mat),columns=['acc1','acc5','precision_macro','recall_macro',
        #                                                          'f1_score_macro','FAR_micro','FRR_micro','FAR_macro','FRR_macro'])
        res = metrics(acc5, con_mat)

    return res