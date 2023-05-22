from Main.FixDateSim import TwoFoldData
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from Core.TrainAndTest import Training,train_mixup
# from NeuralMain import Accuracy
import torch.nn.functional as F
import torch
from Core.utils import config
cfg = config()
savepath= cfg.savepath
data_dir = cfg.data_path

def Accuracy(score,label_train,label_test):
    from Core.utils import MatchID
    pre_ind_nai, pre_ind_gra,_,_= MatchID(score)
    y = np.arange(1,1+score.shape[0]) #The index of label is userid, the minimum value is 1
    naiAcc = sum(label_train.loc[pre_ind_nai+1].values==label_test.loc[y].values)/len(y)
    graAcc = sum(label_test.loc[pre_ind_gra[0]+1].values==label_train.loc[pre_ind_gra[1]+1].values)/len(y)
    return naiAcc,graAcc
def Valid_Graph(model,data,a,b):
    # The user space of the training and test sets may be inconsistent
    model.eval()
    total_loss = 0
    total = 0
    # citeration = nn.CrossEntropyLoss()
    predict = np.zeros((a,b))-1 #a=max(validdata.userid),b = len(traindata.userid)
    for i,(X,y) in enumerate(data):
        y_hat = model(X)
        # error = citeration(y_hat,y)
        # total_loss = total_loss+error.item()
        # save the predict label
        y_hat = F.softmax(y_hat,dim=1)
        predict[y,:] = y_hat.detach().numpy()
        total = total+1
    # total_loss = total_loss/total
    # Delete a line where it is all -1
    predict = predict[np.all(predict>-1,axis=1),:]
    return predict,total_loss
def Test_Graph(model,data,classes):
    model.eval()
    predict = np.zeros((classes,classes))-1
    for i, (X, y) in enumerate(data):
        y_hat = model(X)
        y_hat = F.softmax(y_hat,dim=1)
        predict[y,:] = y_hat.detach().numpy()
    # Delete a line where it is all -1
    predict = predict[np.all(predict>-1,axis=1),:]
    return predict
def TrainUser( data_path = data_dir,data_name ='IPTV.txt',date = [0,16,31]):
    data = pd.read_csv(data_path + data_name, sep='\t', dtype=int, names=['id', 'Item', 'time']);  # index_col=0,that means the index is the value of first column
    data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int')
    TrainData = data[(data['date'] >= date[0]) & (data['date'] <= date[1])]
    user = np.unique(TrainData['id'])
    return user
def NN(tra,tes,cfg,mixup=True,modelname='CNN',havemodel = False,savepath='',savename='',save=False):
    from Core.setRandomSeed import set_random_seed
    set_random_seed(cfg.seed)
    print('Fixed random seed')
    from Core.TrainSetAndTestSet import PrePareData
    Feature_train,label_train,Feature_test,label_test,_ = PrePareData(tra,tes,cfg)
    Classes = Feature_train.shape[0]
    # Classes = max(label_train)
    # max_test_class= max(label_test)
    Inputsize = Feature_train.shape[1]
    # Since train and test may not be the same sample space, the real id needs to be recorded
    train_idx = pd.Categorical(label_train).codes+1
    test_idx = pd.Categorical(label_test).codes+1
    A = pd.Series(label_train,index=train_idx)
    B = pd.Series(label_test,index=test_idx)
    max_test_class = max(test_idx)

    #
    Loss = np.zeros((cfg.epochs,3))
    Acc = np.zeros((cfg.epochs,2,2))

    from Core.TrainAndTest import Test_Graph
    from Core.TrainSetAndTestSet import NN_Input
    from Core.Mymodel import CNNClassifier,MLPClassifier


    Input = DataLoader(NN_Input(Feature_train, train_idx), batch_size=cfg.batch_size, shuffle=True)
    Input_test = DataLoader(NN_Input(Feature_test, test_idx), batch_size=cfg.batch_size, shuffle=False)
    if havemodel==False:
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
        for epoch in range(cfg.epochs):
            model, Loss[epoch,0] = Training(model, Input, optimizer)
            if mixup:
                model, Loss[epoch,1] = train_mixup(model, Input, optimizer, cfg.alpha,getLoss=True)
            Acc[epoch,0,0],Acc[epoch,1,0] = Accuracy(Test_Graph(model,Input,Classes),A,A)
            predict_score,Loss[epoch,2] = Valid_Graph(model,Input_test,max_test_class,Classes) #注意这里的loss在train和test标签空间不一致时，不能用
            Acc[epoch,0,1],Acc[epoch,1,1] =Accuracy(predict_score,A,B)
            # # Acc[epoch,0,1],Acc[epoch,1,1] = Accuracy(Test_Graph(model,Input2,Classes))
            # if (epoch + 1) % 1 == 0:
            #     print('epoch: %d ----- train_loss: %.3f------ train_acc:%.4f----- valid_loss: %.3f------valid_acc:%.4f'
            #           % (epoch + 1, Loss[epoch,0], Acc[epoch,0,0],Loss[epoch,2], Acc[epoch,0,1]))
            scheduler.step()
        if save:
            torch.save(model.state_dict(),savepath+'models\\'+savename+".pth")
        # np.save(savepath+'loss\\'+savename,Loss)
        # np.save(savepath+'acc\\'+savename,Acc)
        # predict_score, Loss[-1, 2] = Valid_Graph(model, Input_test, Classes)  #
        # Acc[-1, 0, 1], Acc[-1, 1, 1] = Accuracy(predict_score)
    else:
        if modelname=='CNN':
            model = CNNClassifier(inputsize=Inputsize,Classes=Classes,
                                  num_kernels=4,pooling_stride=2,kernel_size = 3)
        elif modelname=='MLP':
            model = MLPClassifier(NeuralNum=[Inputsize,512,512,Classes])
        model.load_state_dict(torch.load(savepath+'models\\'+savename+".pth"))
        # Acc[-1,0,0],Acc[-1,1,0] = Accuracy(Test_Graph(model,Input,Classes),A,A)
        predict_score,Loss[-1,2] = Valid_Graph(model,Input_test,max_test_class,Classes) #Note that the loss here cannot be used when the train and test label spaces are inconsistent
        Acc[-1,0,1],Acc[-1,1,1] =Accuracy(predict_score,A,B)
    return Loss,Acc
def FixKnown(cfg,split_date=4):
    subfile = cfg.data_name[:-4]
    savename = subfile+'\\'+subfile \
               +'U'+str(cfg.Num) \
               +'n'+str(cfg.nGram)+'r'+str(cfg.ratio)+'FixknownDate'+str(split_date)
    # from DataProcessing import TwoFoldData
    user = TrainUser(data_name=cfg.data_name,date=[0,split_date,31])
    if cfg.Num>len(user):
        cfg.Num = len(user)
    userID = user[:cfg.Num]
    x = [i for i in range(split_date+1,32)]
    UMISAcc = np.zeros((2,4,len(x)))
    for I in range(len(x)):
        if I==0:
            havemodel = False
        else:
            havemodel = True
        cfg.date = [0,split_date,x[I]]
        TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,
                                             data_name=cfg.data_name,date=cfg.date)
        loss,acc = NN(TrainData,TestData,cfg,mixup=False,modelname='MLP',havemodel=havemodel,
                      savepath=savepath,savename=savename+'MLP',save=True)
        UMISAcc[:,0,I] = acc[-1,:,-1]
        loss,acc = NN(TrainData,TestData,cfg,mixup=True,modelname='MLP',havemodel=havemodel,
                      savepath=savepath,savename=savename+'MLPMix',save=True)
        UMISAcc[:,1,I] = acc[-1,:,-1]
        loss,acc = NN(TrainData,TestData,cfg,mixup=False,modelname='CNN',havemodel=havemodel,
                      savepath=savepath,savename=savename+'CNN',save=True)
        UMISAcc[:,2,I] = acc[-1,:,-1]
        loss,acc = NN(TrainData,TestData,cfg,mixup=True,modelname='CNN',havemodel=havemodel,
                      savepath=savepath,savename=savename+'CNNMix',save=True)
        UMISAcc[:,3,I] = acc[-1,:,-1]
        np.save(savepath+'accs\\'+savename+"AccNN.npy",UMISAcc)
        print(UMISAcc[0,:,I])
    print('n=%d,r=%.2f'%(cfg.nGram,cfg.ratio))
    print('FixKnown,NM-MLP=',UMISAcc[0,0,:])
    print('FixKnown,NM-MLPMix=',UMISAcc[0,1,:])
    print('FixKnown,NM-CNN=',UMISAcc[0,2,:])
    print('FixKnown,NM-CNNMix=',UMISAcc[0,3,:])
    # return UMISAcc
def FixUnknown(cfg,start_date=26):
    subfile = cfg.data_name[:-4]
    savename = subfile+'\\'+subfile \
               +'U'+str(cfg.Num) \
               +'n'+str(cfg.nGram)+'r'+str(cfg.ratio)+'FixunknownDate'+str(start_date+1)
    # from DataProcessing import TwoFoldData
    userID = range(1,cfg.Num+1,1)
    x = [i for i in range(start_date,-1,-1)]
    UMISAcc = np.zeros((2,4,len(x)))
    a = 0
    for I in range(a,len(x)):
        cfg.date = [x[I],start_date,31]
        TrainData, TestData, _ = TwoFoldData(userID=userID,data_path=cfg.data_path,
                                             data_name=cfg.data_name,date=cfg.date)
        loss,acc = NN(TrainData,TestData,cfg,mixup=False,modelname='MLP')
        UMISAcc[:,0,I] = acc[-1,:,-1]
        loss,acc = NN(TrainData,TestData,cfg,mixup=True,modelname='MLP')
        UMISAcc[:,1,I] = acc[-1,:,-1]
        loss,acc = NN(TrainData,TestData,cfg,mixup=False,modelname='CNN')
        UMISAcc[:,2,I] = acc[-1,:,-1]
        loss,acc = NN(TrainData,TestData,cfg,mixup=True,modelname='CNN')
        UMISAcc[:,3,I] = acc[-1,:,-1]
        np.save(savepath+'accs\\'+savename+"AccNN.npy",UMISAcc)
    print('n=%d,r=%.2f'%(cfg.nGram,cfg.ratio))
    print('FixKnown,NM-MLP=',UMISAcc[0,0,:])
    print('FixKnown,NM-MLPMix=',UMISAcc[0,1,:])
    print('FixKnown,NM-CNN=',UMISAcc[0,2,:])
    print('FixKnown,NM-CNNMix=',UMISAcc[0,3,:])
    # return UMISAcc
if __name__ == '__main__':
    cfg.nGram =1
    cfg.ratio=1
    cfg.data_name = 'Shop.txt'
    cfg.ph=4
    cfg.epochs = 400
    cfg.Num= 300
    FixKnown(cfg,split_date=10)
    FixUnknown(cfg,start_date=20)



