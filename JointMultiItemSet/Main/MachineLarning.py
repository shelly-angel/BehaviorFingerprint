import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from Core.utils import MatchID

def norm(X):
    m = X.shape[0]
    return (X-X.mean(axis=1).reshape(m,-1))/X.std(axis=1).reshape(m,-1)
def model_select(X_train,y_train):
    X_train = norm(X_train)
    from sklearn.model_selection import GridSearchCV
    #1.KNN
    # knn = KNeighborsClassifier()
    # grid_param = { 'n_neighbors' : list(range(1,3)) ,
    #                'algorithm' : ['auto','ball_tree','kd_tree','brute']
    #                }
    # rand_ser = GridSearchCV(knn,grid_param,cv = 2)

    #2.SVM
    # svc = SVC() #The result is：0.1，linear,1 , 75.45, The parameters are the same no matter how
    # grid_param = {
    #     'C': [0.1,0.5,1,1.5,2],
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #     'degree': [1,2,3],
    # }
    # rand_ser = GridSearchCV(svc,grid_param,cv = 2)

    svc = LinearSVC(max_iter=2000)  # 'l2','squared_hinge',0.1
    grid_param = {
        'penalty': ['l2'], #'l1',
        'loss':['hinge','squared_hinge'], #'hinge',
        'C':[0.1,0.3,0.5,0.8,1,2,3],
        # 'C':[0.001,0.005,0.01,0.05,0.1]

    }
    rand_ser = GridSearchCV(svc,grid_param,cv = 2)

    # svc = LinearSVC(penalty='l2',loss='squared_hinge',C=0.1)  # 'l2','squared_hinge',0.1
    # grid_param = {
    #     'max_iter':[1000,2000,5000]
    # }
    # rand_ser = GridSearchCV(svc,grid_param,cv = 2)

    # svc = NuSVC() #The result is：0.1，linear,2 , 75.45
    # grid_param = {
    #     'nu': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    #     'kernel':['linear','poly','rbf'],
    #     'degree':[2,3,4]
    # }
    # rand_ser = GridSearchCV(svc,grid_param,cv = 2)

    # # 3. XGboost
    # from xgboost.sklearn import XGBClassifier
    # xgboost = XGBClassifier(objective='multi:softmax',num_class=100)
    # grid_param = {
    #     'learning_rate':[0.05,0.1,0.2,0.3],
    #     'max_depth':[i for i in range(3,10,2)],
    #     'min_child_weight':[i for i in range(1,6,2)] # The results for these parameters are not good
    # }
    # rand_ser = GridSearchCV(xgboost,grid_param,cv = 2)

    # X_train = StandardScaler().fit_transform(X_train)
    rand_ser.fit(X_train,y_train)
    #best parameter combination
    print(rand_ser.best_params_)
    #score achieved with best parameter combination
    print(rand_ser.best_score_)
    #all combinations of hyperparameters
    print(rand_ser.cv_results_['params'])
    #average scores of cross-validation
    print(rand_ser.cv_results_['mean_test_score'])
def test(cfg):
    from Core.setRandomSeed import set_random_seed  # Fixed randomization seed
    set_random_seed(cfg.seed)
    from Core.DataProcessing import Alluser
    userID = Alluser(data_name=cfg.data_name)
    cfg.Num = len(userID)
    from Core.TrainSetAndTestSet import GetFeature
    FeatureMap1,label1,FeatureMap2,label2, \
    _,X,Y,_ \
        = GetFeature(cfg,userID,Test=True,m=cfg.ph)
    X_train = np.r_[FeatureMap1,FeatureMap2];
    y_train = np.r_[label1,label2]
    model_select(X_train,y_train)


def MachineLearning(X_train,y_train,X,Y,method = 'KNN'):
    # # Input normalization
    # scaler = StandardScaler()
    # X_train_norm = scaler.fit_transform(X_train)
    # X_norm = scaler.transform(X)

    X_train_norm = norm(X_train)
    X_norm = norm(X)

    # Model training and testing
    # 1.KNN
    if method=='KNN':
        print('KNN:')
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train_norm,y_train)
        sim = knn.predict_proba(X_norm) # It doesn't return the probability of the distance, it returns the ratio of the nearest neighbor, and if the neighbor here is 1, it's either going to be 1 or 0
        # sim = knn.kneighbors_graph(X_norm,n_neighbors=len(y_train),mode='connectivity')
        # print(knn.score(X_norm,Y))
        pre_ind_nai,pre_ind_gra,_,_ = MatchID(sim)
        nai = sum(Y == y_train[pre_ind_nai]) / len(Y)
        gra = sum(pre_ind_gra[0] == pre_ind_gra[1]) / len(Y)
        print('Classification error rate of KNN(NM,GM)=',1-nai,1-gra)
    #2. SVM
    # ####
    # print('SVC:')
    # svc = SVC(probability=True)
    # svc.fit(X_train_norm,y_train)
    # # print(svc.score(X_norm,Y))
    # # sim = svc.decision_function(X_norm)
    # sim = svc.predict_proba(X_norm)
    # pre_ind_nai,pre_ind_gra,_,_ = MatchID(sim)
    # nai = sum(Y == Y[pre_ind_nai]) / len(Y)
    # gra = sum(pre_ind_gra[0] == pre_ind_gra[1]) / len(Y)
    # print('Classification error rate of SVC(NM,GM)=',1-nai,1-gra)
    # ####
    # print('NuSVC:')
    # svc = NuSVC(nu,max_iter=2000,probability=True)
    # svc.fit(X_train_norm,y_train)
    # # print(svc.score(X1,Y))
    # # sim = svc.decision_function(X1)
    # sim = svc.predict_proba(X_norm)
    # pre_ind_nai,pre_ind_gra,_,_ = MatchID(sim)
    # nai = sum(Y == Y[pre_ind_nai]) / len(Y)
    # gra = sum(pre_ind_gra[0] == pre_ind_gra[1]) / len(Y)
    # print('Classification error rate of NuSVC(NM,GM)=',1-nai,1-gra)
    ####
    # svc = SVC(kernel = 'linear',probability=True)
    # svc.fit(X_train_norm,y_train)
    # # print(svc.score(X1,Y))
    # # sim = svc.decision_function(X1)
    # sim = svc.predict_proba(X_norm)
    # pre_ind_nai,pre_ind_gra,_,_ = MatchID(sim)
    # nai = sum(Y == Y[pre_ind_nai]) / len(Y)
    # gra = sum(pre_ind_gra[0] == pre_ind_gra[1]) / len(Y)
    # print('Classification error rate of LinearSVC(NM,GM)=',1-nai,1-gra)
    elif method=='SVM':
        print('LinearSVC')
        # svc = LinearSVC(penalty='l2',loss='squared_hinge',C=0.1,max_iter=2000)
        svc = LinearSVC(penalty='l2',loss='hinge',C=0.1,max_iter=3000)
        svc.fit(X_train_norm,y_train)
        nai = svc.score(X_norm,Y)
        sim = svc.decision_function(X_norm)
        pre_ind_nai,pre_ind_gra,_,_ = MatchID(sim)
        print(nai,sum(Y == Y[pre_ind_nai]) / len(Y))
        gra = sum(pre_ind_gra[0] == pre_ind_gra[1]) / len(Y)
        print('Classification error rate of LinearSVC(NM,GM)=',1-nai,1-gra)
    elif method=='SVM2':
        print('LinearSVC')
        svc = LinearSVC(penalty='l2',loss='squared_hinge',C=0.1,max_iter=2000)
        # svc = LinearSVC(penalty='l2',loss='hinge',C=0.1,max_iter=3000)
        svc.fit(X_train_norm,y_train)
        nai = svc.score(X_norm,Y)
        sim = svc.decision_function(X_norm)
        pre_ind_nai,pre_ind_gra,_,_ = MatchID(sim)
        print(nai,sum(Y == Y[pre_ind_nai]) / len(Y))
        gra = sum(pre_ind_gra[0] == pre_ind_gra[1]) / len(Y)
        print('Classification error rate of LinearSVC(NM,GM)=',1-nai,1-gra)
    # #3. XGBoost
    # print('XGBoost')
    # from xgboost.sklearn import XGBClassifier
    # xgboost = XGBClassifier(num_class=cfg.end,reg_alpha=1)
    # xgboost.fit(X_train,y_train)
    # print(xgboost)
    # sim = xgboost.predict_proba(X)
    # pre_ind_nai,pre_ind_gra,_,_ = MatchID(sim)
    # nai = sum(Y == Y[pre_ind_nai]) / len(Y)
    # gra = sum(pre_ind_gra[0] == pre_ind_gra[1]) / len(Y)
    # print('Classification error rate of XGBoost(NM,GM)=',1-nai,1-gra)
    #4. LDA
    # print('LDA')
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    # lda = LDA()
    return nai,gra
def main(cfg,m):
    from Core.setRandomSeed import set_random_seed  # Fixed randomization seed
    set_random_seed(cfg.seed)
    from Core.DataProcessing import Alluser
    userID = Alluser(data_name=cfg.data_name)
    cfg.Num = len(userID)
    from Core.TrainSetAndTestSet import GetFeature
    FeatureMap1,label1,FeatureMap2,label2, \
    _,X,Y,_ \
        = GetFeature(cfg,userID,Test=True,m=cfg.ph)
    X_train = np.r_[FeatureMap1,FeatureMap2]
    y_train = np.r_[label1,label2]
    nai,gra = MachineLearning(X_train,y_train,X,Y,method=m)
    return nai,gra
def f(cfg,name):
    Acc = np.zeros((2,2,1))
    for s in range(1):
        cfg.seed = s
        Acc[0,0,s],Acc[0,1,s] = main(cfg,m='KNN')
        Acc[1,0,s],Acc[1,1,s] = main(cfg,m ='SVM')
        # Acc[0,0,s],Acc[0,1,s] = main(cfg,m='SVM2')
        np.save(cfg.savepath+'accs\\'+name+'_ML.npy',Acc)
        print(Acc[:,:,s])
    print('KNN')
    print(np.mean(1-Acc[0,:,:],axis = 1))
    print(np.std(1-Acc[0,:,:],axis = 1))
    print('SVM')
    print(np.mean(1-Acc[1,:,:],axis = 1))
    print(np.std(1-Acc[1,:,:],axis = 1))
    # return Acc

if __name__ == '__main__':
    """
    It takes a long time, especially for the SVM part. 
    It is suggested to run one section at a time or store the results
    """

    from Core.utils import config
    cfg = config()
    # #
    # cfg.Num = 1000
    # cfg.nGram = 1
    # cfg.ratio = 1
    # cfg.ph = 3
    # # main(cfg,nu=0.1) #The default is IPTV data set
    # # test(cfg)
    # # f(cfg,name='IPTVU1000N1R1')
    #
    #
    # # #
    # cfg.Num = 299 #All users of the entire data set
    # cfg.nGram = 1
    # cfg.ratio = 1
    # cfg.ph = 4
    # cfg.data_name = "Shop.txt"
    # # # main(cfg,nu=0.8)
    # # # test(cfg)
    # # f(cfg,name='ShopU299N1R1')
    # # # #
    cfg.Num = 945 #All users of the entire data set
    cfg.nGram = 1
    cfg.ratio = 1
    cfg.ph = 4
    cfg.data_name = "Reddit.txt"
    # test(cfg)
    # # main(cfg,nu=0.8)
    f(cfg,name='RedditU945N1R1')
    # # #
    # # # #
    # # cfg.end = 1000
    # # cfg.nGram = 2
    # # cfg.ratio = 0.2
    # # cfg.ph=3
    # # cfg.data_name = "IPTV.txt"
    # # # main(cfg,nu=0.1)
    # # f(cfg,name='IPTVU1000N2R0.2')
    # # #
    # # # #
    # # # # cfg.end = 299
    # # # # cfg.nGram = 2
    # # # # cfg.ratio = 0.2
    # # # # cfg.data_name = "Shop.txt"
    # # # # main(cfg,m=4,nu=0.8)
    # # #
    # # # #
    # # # # cfg.end = 945
    # # # # cfg.nGram = 2
    # # # # cfg.ratio = 0.2
    # # # # cfg.ph = 4
    # # # # cfg.data_name = "Reddit.txt" # C=0.1,loss=hinge,penalty='l2'  这个实在是跑得特别慢
    # # # # main(cfg,nu=0.8)
