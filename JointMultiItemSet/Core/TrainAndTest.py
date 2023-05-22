import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F

def Training(model,data,optimizer):
    model.train()
    total_loss = 0;
    total = 0
    citeration = nn.CrossEntropyLoss()
    for i,(X,y) in enumerate(data):
        # Gradient zeroing
        model.zero_grad()
        # forward computation
        y_hat = model.forward(X)
        # Error between prediction and reality
        error = citeration(y_hat, y)
        error.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # prevent gradient explosion
        # Gradient update
        optimizer.step()
        total += 1
        total_loss += error.item()
    total_loss = total_loss/total
    return model,total_loss
def Valid(model,data):
    model.eval()
    total_loss = 0
    total = 0
    citeration = nn.CrossEntropyLoss()
    predict_label =  pd.DataFrame(columns=['y_hat', 'y'])
    for i,(X,y) in enumerate(data):
        y_hat = model(X)
        error = citeration(y_hat,y)
        total_loss = total_loss+error.item()
        # save the predict label
        y_hat = torch.max(y_hat, 1)[1].numpy()
        y = y.numpy()
        rows = len(y)
        total += 1
        for J in range(rows):
            predict_label.loc[i * rows + J] = [y_hat[J], y[J]]
    predict_label.index = predict_label['y']
    total_loss = total_loss/total
    return predict_label,total_loss
def Test(model,data):
    model.eval()
    predict_label = pd.DataFrame(columns=['y_hat', 'y'])
    for i,(X,y) in enumerate(data):
        y_hat = model(X)
        # save the predict label
        y_hat = torch.max(y_hat, 1)[1].numpy()
        y = y.numpy()
        rows = len(y)
        for J in range(rows):
            predict_label.loc[i*rows+J] = [y_hat[J],y[J]]
    predict_label.index = predict_label['y']
    return predict_label
def metric(Yhat_Y):
    acc = sum(Yhat_Y['y']==Yhat_Y['y_hat'])/len(Yhat_Y['y'])
    return acc
def Valid_Graph(model,data,classes):
    model.eval()
    total_loss = 0
    total = 0
    citeration = nn.CrossEntropyLoss()
    predict = np.zeros((classes,classes))
    for i,(X,y) in enumerate(data):
        y_hat = model(X)
        error = citeration(y_hat,y)
        total_loss = total_loss+error.item()
        # save the predict label
        y_hat = F.softmax(y_hat,dim=1)
        predict[y,:] = y_hat.detach().numpy()
        total = total+1
    total_loss = total_loss/total
    return predict,total_loss
def Test_Graph(model,data,classes):
    model.eval()
    predict = np.zeros((classes,classes))
    for i, (X, y) in enumerate(data):
        y_hat = model(X)
        y_hat = F.softmax(y_hat,dim=1)
        predict[y,:] = y_hat.detach().numpy()
    return predict

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x,y_a,y_b,lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_mixup(net,trainloader,optimizer,alpha,getLoss=False):
    net.train()
    train_loss = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,alpha)
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        total += targets.size(0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)  # prevent gradient explosion
        optimizer.step()
    if getLoss:
        return net,train_loss
    else:
        return net

def Learner(FeatureMap,label,FeatureNum,Classes,cfg,mixup=True,modelname='CNN'):
    from Core.Mymodel import CNNClassifier,MLPClassifier
    if modelname=='CNN':
        model = CNNClassifier(inputsize=FeatureNum,Classes=Classes,
                                  num_kernels=4,pooling_stride=2,kernel_size = 3)
    elif modelname=='MLP':
        model = MLPClassifier(NeuralNum=[FeatureNum,512,512,Classes])
    else:
        assert 1==2,'The structure of the model needs to be set.'
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    from torch.utils.data import DataLoader
    from Core.TrainSetAndTestSet import NN_Input
    Input = DataLoader(NN_Input(FeatureMap, label), batch_size=cfg.batch_size, shuffle=True)
    Loss = np.zeros((cfg.epochs,))
    Acc = np.zeros((cfg.epochs))
    if mixup:
        for epoch in range(cfg.epochs):
            model, Loss[epoch] = Training(model, Input, optimizer)
            model = train_mixup(model, Input, optimizer, cfg.alpha)
            # Acc[epoch] = metric(Test(model, Input))
            scheduler.step()  # Adjust the learning rate
            # if (epoch + 1) % 100 == 0:
            #     print('epoch: %d ----- train_loss: %.3f------ train_acc:%.4f'
            #           % (epoch + 1,Loss[epoch],Acc[epoch]))
    else:
        for epoch in range(cfg.epochs):
            model, Loss[epoch] = Training(model, Input, optimizer)
            scheduler.step()  # Adjust the learning rate
    return model