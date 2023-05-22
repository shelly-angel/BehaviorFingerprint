import numpy as np
from torch.utils.data import DataLoader
from Core.TrainAndTest import Training,train_mixup
from Core.NeuralMain import Accuracy

def NN(tra,tes,cfg,mixup=True,modelname='CNN',savepath='',savename=''):
    from Core.setRandomSeed import set_random_seed
    set_random_seed(cfg.seed)
    print('Fixed random seed')
    from Core.TrainSetAndTestSet import PrePareData
    Feature_train,label_train,Feature_test,label_test,_ = PrePareData(tra,tes,cfg)
    Classes = Feature_train.shape[0]
    Inputsize = Feature_train.shape[1]
    from Core.TrainAndTest import Valid_Graph
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
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    Input = DataLoader(NN_Input(Feature_train, label_train), batch_size=cfg.batch_size, shuffle=True)
    Input_test = DataLoader(NN_Input(Feature_test, label_test), batch_size=cfg.batch_size, shuffle=False)
    #
    Loss = np.zeros((cfg.epochs,3))
    Acc = np.zeros((cfg.epochs,2,2))
    for epoch in range(cfg.epochs):
        model, Loss[epoch,0] = Training(model, Input, optimizer)
        if mixup:
            model, Loss[epoch,1] = train_mixup(model, Input, optimizer, cfg.alpha,getLoss=True)
        # Acc[epoch,0,0],Acc[epoch,1,0] = Accuracy(Test_Graph(model,Input,Classes))
        # predict_score,Loss[epoch,2] = Valid_Graph(model,Input_test,Classes) #
        # Acc[epoch,0,1],Acc[epoch,1,1] =Accuracy(predict_score)
        # # Acc[epoch,0,1],Acc[epoch,1,1] = Accuracy(Test_Graph(model,Input2,Classes))
        # if (epoch + 1) % 100 == 0:
        #     print('epoch: %d ----- train_loss: %.3f------ train_acc:%.4f----- valid_loss: %.3f------valid_acc:%.4f'
        #           % (epoch + 1, Loss[epoch,0], Acc[epoch,0,0],Loss[epoch,2], Acc[epoch,0,1]))
        scheduler.step()
    # torch.save(model.state_dict(),savepath+'models\\'+savename+".pth")
    # np.save(savepath+'loss\\'+savename,Loss)
    # np.save(savepath+'acc\\'+savename,Acc)
    predict_score,Loss[-1,2] = Valid_Graph(model,Input_test,Classes) #
    Acc[-1,0,1],Acc[-1,1,1] =Accuracy(predict_score)
    return Loss,Acc

def seed(cfg):
    seed = range(10)
    subfile = cfg.data_name[:-4]
    savename = subfile+'\\'+subfile \
               +'U'+str(cfg.Num)+'n'+str(cfg.nGram)+'r'+str(cfg.ratio) \
               +'splitdate'+str(cfg.date[-2])+'seed'+str(seed[0])+'_'+str(seed[-1])+'.npy'
    from Core.DataProcessing import CommonUser,TwoFoldData
    user = CommonUser(data_path=cfg.data_path, data_name=cfg.data_name,date=cfg.date)
    if len(user)<cfg.Num:
        cfg.Num = len(user)
    userID = user[:cfg.Num]
    TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,data_name=cfg.data_name,date=cfg.date)
    # Loss = np.zeros((len(seed),cfg.epochs,3))
    Acc = np.zeros((4,2,len(seed)))
    I = 0
    for a in seed:
        cfg.seed = a
        loss,acc = NN(TrainData,TestData,cfg,mixup=False,modelname='MLP')
        Acc[0,:,I] = acc[-1,:,-1]
        loss,acc = NN(TrainData,TestData,cfg,mixup=True,modelname='MLP')
        Acc[1,:,I] = acc[-1,:,-1]
        loss,acc = NN(TrainData,TestData,cfg,mixup=False,modelname='CNN')
        Acc[2,:,I] = acc[-1,:,-1]
        loss,acc = NN(TrainData,TestData,cfg,mixup=True,modelname='CNN')
        Acc[3,:,I] = acc[-1,:,-1]
        np.save(cfg.savepath+'accs\\'+savename,Acc)
        I = I+1
    print('NM-MLP,MLPMix,CNN,CNNMix',np.mean(Acc[:,0,:],axis=1))
    print('GM-MLP,MLPMix,CNN,CNNMix',np.mean(Acc[:,1,:],axis=1))
if __name__ == '__main__':
    from Core.utils import config
    cfg = config()

    cfg.data_name = "IPTV.txt"
    cfg.nGram = 1
    cfg.ratio = 1
    cfg.epochs = 150
    cfg.Num = 1000
    seed(cfg)

    cfg.nGram =1
    cfg.ratio=1
    cfg.Num = 300
    cfg.data_name = 'Shop.txt'
    cfg.ph=4
    cfg.epochs = 400
    seed(cfg)

    cfg.nGram=1
    cfg.ratio=1
    cfg.Num = 1000
    cfg.data_name = 'Reddit.txt'
    cfg.ph=4
    cfg.epochs = 400
    seed(cfg)