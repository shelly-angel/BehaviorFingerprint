import numpy as np
from torch.utils.data import DataLoader
def AssignWeights(Sim_Tensor,knownUser,unknownUser,Graph=False):
    # Sim_Tensor = (Classes,len(methods),Classes) ,and len(knownUser)==unknoenUser ==Classes
    Classes,len_methods,_ = Sim_Tensor.shape
    weights = np.zeros((len_methods,))
    if Graph:
        from scipy.optimize import linear_sum_assignment
        for I in range(len_methods):
            sim = Sim_Tensor[:,I,:]
            # np.nan_to_num(sim, copy=False)
            predict_label_index = linear_sum_assignment(sim, maximize=True)
            weights[I] = sum(unknownUser[predict_label_index[0]] == knownUser[predict_label_index[1]]) / Classes
    else:
        for I in range(len_methods):
            sim = Sim_Tensor[:,I,:]
            # np.nan_to_num(sim, copy=False)
            predict_label_index = sim.argmax(axis=1)
            weights[I] = sum(unknownUser == knownUser[predict_label_index]) / Classes
    return weights
def Score(ftra,ftes,label,K,model,Classes,Classic = True):
    Second_Layer_input = np.zeros((Classes,K,Classes))
    if Classic:
        from scipy.spatial.distance import cdist
        # Jaccard
        Second_Layer_input[:,0,:] = 1 - cdist(np.array(ftes, dtype=bool), np.array(ftra, dtype=bool), 'jaccard')
        # KL
        Second_Layer_input[:,1,:] = np.square(cdist(ftes,ftra,'jensenshannon'))*2
        # JKL
        Second_Layer_input[:,2,:] = Second_Layer_input[:,0,:]/(Second_Layer_input[:,1,:]+1e-6)
        # KL needs a change
        Second_Layer_input[:,1,:] = 2 - Second_Layer_input[:,1,:]
    # NN
    from Core.TrainAndTest import Test_Graph
    from Core.TrainSetAndTestSet import NN_Input
    Input = DataLoader(NN_Input(ftes, label), batch_size=1, shuffle=False)
    Second_Layer_input[:,K-1,:] = Test_Graph(model,Input,Classes)
    return Second_Layer_input
def SingleMatcherAcc(sim,knownUser,unknownUser):
    K = sim.shape[1]
    Acc = np.zeros((2,K))
    np.nan_to_num(sim,copy=False)
    from scipy.optimize import linear_sum_assignment
    for I in range(K):
        predict_label_index = sim[:, I, :].argmax(axis=1)
        Acc[0, I] = sum(unknownUser == knownUser[predict_label_index]) / len(unknownUser)
        predict_label_index = linear_sum_assignment(sim[:, I, :], maximize=True)
        Acc[1, I] = sum(unknownUser[predict_label_index[0]] == knownUser[predict_label_index[1]]) / len(unknownUser)
    return Acc
def fusionAcc(sim,knownUser,unknownUser,weights_naive,weights_graph):
    K = sim.shape[1]
    C = len(unknownUser)
    Acc = np.zeros((2, 1))
    np.nan_to_num(sim, copy=False)
    from scipy.optimize import linear_sum_assignment
    weights_graph = weights_graph/np.sum(weights_graph)
    weights_naive = weights_naive/np.sum(weights_naive)
    # Normalize KL and JKL
    sim[:,1,:] = sim[:,1,:]/2
    JKL_min = np.min(sim[:,2,:]);JKL_max=np.max(sim[:,2,:])
    sim[:,2,:] = (sim[:,2,:] -JKL_min)/(JKL_max-JKL_min)
    # Calculate the identification accuracy of fusion results
    temp = np.zeros((C, C))
    for I in range(K):
        temp = weights_naive[I] * sim[:, I, :] + temp
    np.nan_to_num(temp, copy=False)
    predict_label_index = temp.argmax(axis=1)
    Acc[0] = sum(unknownUser== knownUser[predict_label_index]) / C
    temp = np.zeros((C, C))
    for I in range(K):
        temp = weights_graph[I] * sim[:, I, :] + temp
    np.nan_to_num(temp, copy=False)
    predict_label_index = linear_sum_assignment(temp, maximize=True)
    Acc[1] = sum(unknownUser[predict_label_index[0]] == knownUser[predict_label_index[1]]) / C
    return Acc
def FusionScore(sim,weights_naive,C,K=4):
    np.nan_to_num(sim, copy=False)
    weights_naive = weights_naive / np.sum(weights_naive)
    # Normalize KL and JKL
    sim[:, 1, :] = sim[:, 1, :] / 2
    JKL_min = np.min(sim[:, 2, :]);
    JKL_max = np.max(sim[:, 2, :])
    sim[:, 2, :] = (sim[:, 2, :] - JKL_min) / (JKL_max - JKL_min)
    # Calculate the identification accuracy of fusion results
    temp = np.zeros((C, C))
    for I in range(K):
        temp = weights_naive[I] * sim[:, I, :] + temp
    np.nan_to_num(temp, copy=False)
    return temp

# def main(cfg,seed):
#     from TrainSetAndTestSet import GetFeature
#     FeatureMap1,label1,FeatureMap2,label2,Classes,X,Y,FeatureName = GetFeature(cfg,seed,Test=True)
#     methods = ['Jaccard', 'KL', 'JKL', 'CNN']
#     K = len(methods)
#     # part1
#     from TrainAndTest import Learner
#     model = Learner(FeatureMap1, label1, len(FeatureName), Classes, cfg)
#     Input = Score(FeatureMap1,FeatureMap2,label2,K,model,Classes)
#     weights_naive = AssignWeights(Input,knownUser=label1,unknownUser=label2)
#     weights_graph = AssignWeights(Input,knownUser=label1,unknownUser=label2,Graph=True)
#     # part2
#     model = Learner(FeatureMap2, label2, len(FeatureName), Classes, cfg)
#     Input2 = Score(FeatureMap2,FeatureMap1,label1,K,model,Classes,Classic=False) #相似度不用再算一遍了，转置就好了
#     Input2[:,:(K-1),:] = Input[:,:(K-1),:].transpose(2,1,0) # 之前的需要转置
#     # Average
#     weights_naive = (AssignWeights(Input2,knownUser=label2,unknownUser=label1)+weights_naive)/2
#     weights_graph = (AssignWeights(Input2,knownUser=label1,unknownUser=label2,Graph=True)+weights_graph)/2
#     # part3
#     model = Learner(np.r_[FeatureMap1,FeatureMap2], np.r_[label1,label2],
#                     len(FeatureName), Classes, cfg)
#     Input = Score(FeatureMap1 + FeatureMap2, X,Y,K, model, Classes)
#     Acc = fusionAcc(Input, label1, Y, methods,Classes,weights_naive,weights_graph)
#
#     return Acc
