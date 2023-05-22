# Including loading, dividing data sets, extracting closet users
import numpy as np
import pandas as pd
# Choose User- Random
def RandomChoose(Users = 100,data_path =" ",data_name ='IPTV.txt',date = [0,10,20,31] ):
    data = pd.read_csv(data_path + data_name, sep='\t', dtype=int, names=['id', 'Item', 'time']);
    data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int')
    TrainData = data[(data['date'] >= date[0]) & (data['date'] <= date[1])]
    ValidData = data[(data['date'] > date[1]) & (data['date'] <= date[2])]
    TestData  = data[(data['date'] > date[2]) & (data['date'] <= date[3])]
    #One to one correspondence between known users and anonymous users:
    from functools import reduce
    user = reduce(np.intersect1d,[
        np.unique(TrainData['id']),
        np.unique(ValidData['id']),
        np.unique(TestData['id'])
    ])
    Classes = len(user)
    assert Classes,'there is no same user'
    # assert Classes >= Users,'Out of the range'
    if Classes<Users:
        Users = Classes
        print('Out of the range,The number of users is changed to',Classes)
    if data_name == 'IPTV.txt':
        user = user[:1000]
        # user = user[user<=1000]
    # Randomly select users:
    np.random.shuffle(user)
    user = user[:Users]
    print('Randomly select users and end!')
    print('Number of users: %d'%(len(user)))
    return user
# Choose User- AllUser
def Alluser(data_path = " ",data_name ='IPTV.txt',date = [0,10,20,31] ):
    data = pd.read_csv(data_path + data_name, sep='\t', dtype=int, names=['id', 'Item', 'time']);
    data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int')
    TrainData = data[(data['date'] >= date[0]) & (data['date'] <= date[1])]
    ValidData = data[(data['date'] > date[1]) & (data['date'] <= date[2])]
    TestData  = data[(data['date'] > date[2]) & (data['date'] <= date[3])]
    # One to one correspondence between known users and anonymous users:
    from functools import reduce
    user = reduce(np.intersect1d,[
        np.unique(TrainData['id']),
        np.unique(ValidData['id']),
        np.unique(TestData['id'])
    ])
    Classes = len(user)
    assert Classes,'there is no same user'
    # assert Classes >= Users,'Out of the range'
    if data_name == 'IPTV.txt':
        user = user[:1000]
    print('Number of users: %d'%(len(user)))
    return user
# Get Data- Training set, Valid set, Test set
def ImportData(userID = range(1,1001),data_path = " ",
               data_name ='IPTV.txt',
               date = [0,10,20,31]):
    print('1.Import data from the file: '+data_name) #
    data = pd.read_csv(data_path + data_name,sep='\t', dtype=int, names=['id', 'Item', 'time']);
    data.index = data['id']
    data = data.loc[userID]
    data['id'] = pd.Categorical(data['id']).codes + 1
    data = data.reset_index(drop=True)
    data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int')
    data = data.drop(['time'],axis = 1)
    data['Item'] = pd.Categorical(data['Item']).codes + 1
    print("2.Divide the dataset by dates(train,valid,test)=(%d~%d,%d~%d,%d~%d)"%(date[0],date[1],date[1]+1,date[2],date[2]+1,date[3]))
    TrainData = data[(data['date'] >= date[0]) & (data['date'] <= date[1])]
    ValidData = data[(data['date'] > date[1]) & (data['date'] <= date[2])]
    TestData = data[(data['date'] > date[2]) & (data['date'] <= date[3])]
    Classes = len(np.unique(TrainData['id']))
    return TrainData,ValidData,TestData,Classes

# Choose User
def CommonUser(UsersNum=None,data_path = " ",
                data_name ='IPTV.txt',
                date = [0,16,31]):
    data = pd.read_csv(data_path + data_name, sep='\t', dtype=int, names=['id', 'Item', 'time']);  # index_col=0,that means the index is the value of first column
    data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int') #第几天
    TrainData = data[(data['date'] >= date[0]) & (data['date'] <= date[1])]
    ValidData = data[(data['date'] > date[1]) & (data['date'] <= date[2])]
    # One to one correspondence between known users and anonymous users:
    from functools import reduce
    user = reduce(np.intersect1d,[
        np.unique(TrainData['id']),
        np.unique(ValidData['id']),
    ])
    Classes = len(user)
    assert Classes,'there is no same user'
    if UsersNum==None: return user
    else:
        assert Classes >= UsersNum,'Out of the range'
        user = user[:UsersNum]
        return user
# Get Data- Training Set, Test Set
def TwoFoldData(userID=range(1,1001),data_path = " ",
                data_name ='IPTV.txt',
                date = [0,16,31]):
    print('1.Import data from the file:'+data_name) #
    data = pd.read_csv(data_path + data_name,sep='\t', dtype=int, names=['id', 'Item', 'time'])
    data.index = data['id']
    data = data.loc[userID]
    data['id'] = pd.Categorical(data['id']).codes + 1
    data = data.reset_index(drop=True)
    data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int')
    data = data.drop(['time'],axis = 1)
    data['Item'] = pd.Categorical(data['Item']).codes + 1
    print("2.Divide the dataset by date(train,test)=(%d~%d,%d~%d)"%(date[0],date[1],date[1]+1,date[2]))
    TrainData = data[(data['date'] >= date[0]) & (data['date'] <= date[1])]
    ValidData = data[(data['date'] > date[1]) & (data['date'] <= date[2])]
    #
    Classes = len(userID)
    print('closet userNum is: %d'%(Classes))
    return TrainData,ValidData,Classes

##########################################################################
def ImportDataSet(file,sep='\t',unit='day'):
    if unit=='seconds':
        data = pd.read_csv(file,sep=sep, dtype=int, names=['id', 'Item', 'time'])
        userid = np.unique(data['id'])
        data['id'] = pd.Categorical(data['id']).codes + 1
        data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int')
        data = data.drop(['time'],axis = 1)
    if unit=='day':
        data = pd.read_csv(file,sep=sep, dtype=int, names=['id', 'Item', 'date'])
        userid = np.unique(data['id'])
        data['id'] = pd.Categorical(data['id']).codes + 1
    return data,userid


