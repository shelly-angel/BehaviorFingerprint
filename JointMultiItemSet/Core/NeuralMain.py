import numpy as np
from torch.utils.data import DataLoader
from Core.TrainAndTest import Training,train_mixup

def Accuracy(score):
    from Core.utils import MatchID
    pre_ind_nai, pre_ind_gra,_,_= MatchID(score)
    # The index corresponds to the id, and the accuracy can be calculated directly here, and finally the matching accuracy can be returned.
    y = np.arange(score.shape[0])
    naiAcc = sum(pre_ind_nai==y)/len(y)
    graAcc = sum(pre_ind_gra[0]==pre_ind_gra[1])/len(y)
    return naiAcc,graAcc

def NN(tra,tes,cfg,mixup=True,modelname='CNN',savepath='',savename=''):
    from Core.setRandomSeed import set_random_seed
    set_random_seed(cfg.seed)
    print('Fixed random seed')
    from Core.TrainSetAndTestSet import PrePareData
    Feature_train,label_train,Feature_test,label_test,_ = PrePareData(tra,tes,cfg)
    Classes = Feature_train.shape[0]
    Inputsize = Feature_train.shape[1]
    from Core.TrainAndTest import Valid_Graph,Test_Graph
    from Core.TrainSetAndTestSet import NN_Input
    from Core.Mymodel import CNNClassifier,MLPClassifier
    if modelname=='CNN':
        model = CNNClassifier(inputsize=Inputsize,Classes=Classes,
                              num_kernels=4,pooling_stride=2,kernel_size = 3)
    elif modelname=='MLP':
        model = MLPClassifier(NeuralNum=[Inputsize,512,512,Classes])
    else:
        assert 1==2,'The structure of the model needs to be set.'
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    Input = DataLoader(NN_Input(Feature_train, label_train), batch_size=cfg.batch_size, shuffle=True)
    Input_test = DataLoader(NN_Input(Feature_test, label_test), batch_size=cfg.batch_size, shuffle=False)
    #
    Loss = np.zeros((cfg.epochs,3)) #Loss_real_training，Loss_virtual_Training; Loss_real_valid
    Acc = np.zeros((cfg.epochs,2,2)) #train_NM,train_GM,valid_NM,valid_GM
    for epoch in range(cfg.epochs):
        model, Loss[epoch,0] = Training(model, Input, optimizer)
        if mixup:
            model, Loss[epoch,1] = train_mixup(model, Input, optimizer, cfg.alpha,getLoss=True)
        Acc[epoch,0,0],Acc[epoch,1,0] = Accuracy(Test_Graph(model,Input,Classes))
        predict_score,Loss[epoch,2] = Valid_Graph(model,Input_test,Classes) #
        Acc[epoch,0,1],Acc[epoch,1,1] =Accuracy(predict_score)
        # if (epoch + 1) % 10 == 0:
        #     print('epoch: %d ----- train_loss: %.3f------ train_acc:%.4f----- valid_loss: %.3f------valid_acc:%.4f'
        #           % (epoch + 1, Loss[epoch,0], Acc[epoch,0,0],Loss[epoch,2], Acc[epoch,0,1]))
        scheduler.step()
    # torch.save(model.state_dict(),savepath+'models\\'+savename+".pth")
    # np.save(savepath+'loss\\'+savename,Loss)
    # np.save(savepath+'acc\\'+savename,Acc)
    # return Loss,Acc
    return Acc[-1,:,1] #Only the results of the test set are returned

def earlystop(cnt,loss_a,loss_b):
    if abs(loss_b-loss_a)<=0.05:
        cnt[0]+=1
    else:
        cnt[0]=0
    return cnt[0]>30
def CNNMix(Feature_train,label_train,cfg,Inputsize,Classes):
    from Core.Mymodel import CNNClassifier
    from Core.TrainAndTest import Training, train_mixup
    from Core.TrainAndTest import Test_Graph
    from Core.TrainSetAndTestSet import NN_Input
    model = CNNClassifier(inputsize=Inputsize, Classes=Classes,
                            num_kernels=4, pooling_stride=2, kernel_size=3)
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    Input = DataLoader(NN_Input(Feature_train, label_train), batch_size=cfg.batch_size, shuffle=True)
    for epoch in range(cfg.epochs):
        model, l = Training(model, Input, optimizer)
        model = train_mixup(model, Input, optimizer, cfg.alpha, getLoss=False)
        nm, gm = Accuracy(Test_Graph(model, Input, Classes))
        # if (epoch + 1) % 10 == 0:
        #     print('epoch: %d ----- train_loss: %.3f------ train_acc:%.4f'
        #           % (epoch + 1,l,nm))
        scheduler.step()
    return model
if __name__ == '__main__':
    from Core.utils import config
    cfg = config()
    cfg.date = [0,8,16]
    from Core.DataProcessing import CommonUser,TwoFoldData
    user = CommonUser(data_path=cfg.data_path, data_name=cfg.data_name,date=cfg.date)
    userID = user[:cfg.Num]
    TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,
                                     data_name=cfg.data_name,date=cfg.date)
    loss,acc = NN(tra=TrainData,tes=TestData,mixup=True,modelname='CNN')
