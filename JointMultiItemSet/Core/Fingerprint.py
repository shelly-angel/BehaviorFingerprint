# Construct features from data -Union of Multi-Item-Sets Features
from collections import Counter
import math
import numpy as np
import pandas as pd
### Union Of Multi-Item-Sets Feature
## Feature Space---SURE+
def ChooseFeature(data,Multi=2,ratio = 0.2,max_value_len = None):
    user = np.unique(data['id'])
    data['Item'] = data['Item'].astype(int)
    if max_value_len==None:
        print('If you do not specify the number of digits of the item, problems may arise. The default is the largest number of digits in the training set.')
        # max_value_len = len(str(max(data['Item'])))
        max_value_len = np.log10(max(data['Item']))
    featureName = {}
    for u in user:
        channel = data[data['id'] == u]['Item'].values # The extracted is a numpy array
        length = len(channel)
        Multi_u = min(length,Multi)# Guaranteed to have 'n'-level fingerprints
        feature_count = {}
        for K in range(1, Multi_u+1):
            ch = 0
            for KK in range(0,K):
                ch = ch+((channel[KK:(length-(K-KK-1))])*(10**(max_value_len*KK)))
            feature_count.update(dict(Counter(ch))) #count for each feature
        featureNum =math.ceil(len(feature_count) * ratio)
        featureName_u = dict(sorted(feature_count.items(), key=lambda x: x[1], reverse=True)[:featureNum])
        featureName.update(featureName_u) #Updated by 'keys', the same key and different values will keep the one in the parentheses, but it doesn't matter, because the value is not used here; the first FeatureName here is an empty dictionary
    FeatureName = set(featureName.keys())
    return FeatureName,max_value_len
## Generate feature vector from feature space
def UnionAllFea(data,FeatureName,max_value_len=3,Multi=1):
    user = np.unique(data['id'])
    data['Item'] = data['Item'].astype(int)
    TotalFeature = len(FeatureName)
    Feature = np.zeros((len(user),TotalFeature),'int32')
    Label = np.zeros(len(user),'int32')
    # Establish an index to facilitate the count of feature behaviors to be placed in the corresponding position
    Inx= pd.DataFrame(data=list(FeatureName),columns=['Name'],copy=True)
    Inx['inx'] = np.arange(TotalFeature)
    Inx = Inx.set_index(Inx['Name'])
    Inx.drop('Name',axis=1,inplace=True)
    # Get the feature vector of each user
    J = 0;
    for u in user:
        data_u = data[data['id'] == u]
        channel = data_u['Item'].values
        stas_Matrix = np.zeros((1, TotalFeature), 'int32')
        length = len(channel)
        Multi_u = min(length,Multi)# Guaranteed to have 'n'-level fingerprints
        for K in range(1, Multi_u + 1):
            ch = 0
            for KK in range(0, K):
                ch = ch + ((channel[KK:(length - (K - KK - 1))]) * (10 ** (max_value_len * KK)))
            data_u_group = pd.DataFrame(data=ch,columns=['Item'],copy=True)
            data_u_group = data_u_group.loc[data_u_group['Item'].isin(FeatureName)]
            data_u_group = data_u_group.groupby(by=['Item'])  #
            stas = data_u_group.size()  # Count the number of times each item appears in groups
            stas = stas.reset_index()  # Convert to dataframe
            stas.columns = ['Item','times']
            ChannelIndex = Inx.loc[stas['Item'].values,'inx']
            ChannelIndex = ChannelIndex.values
            stas_Matrix[0, ChannelIndex] = stas['times']
        Feature[J,:] = stas_Matrix;
        Label[J] = u;
        J = J+1;
    return Feature,Label
# Generate feature vector for each day from feature space
def FeatureVector(data,FeatureName,max_value_len=3,Multi=1):
    # Obtain the feature vector of each unit (such as, day) time, the input data format is <userid,itemid(CurChannel),time(s)>
    user = np.unique(data['id'])
    data['Item'] = data['Item'].astype(int)
    # data['date'] = np.floor(data['time'] / (24 * 3600)).astype('int') # day as the base unit
    # interval = np.unique(data.time)
    # samples = len(interval)
    # min_time = np.min(data.time)
    # max_time = np.max(data.time)
    # samples = max_time-min_time+1

    interval = np.unique(data.date)
    samples = len(interval)
    TotalFeature = len(FeatureName)
    Feature = np.zeros((len(user)*samples,2+TotalFeature)) #2+TotalFeature=uid,date,features
    # bulid an index
    Inx= pd.DataFrame(data=list(FeatureName),columns=['Name'],copy=True)
    Inx['inx'] = np.arange(TotalFeature)
    Inx = Inx.set_index(Inx['Name'])
    Inx.drop('Name',axis=1,inplace=True)

    J = 0;
    for u in user:
        data_u = data[data['id']==u]
        for d in interval:
            data_u_d = data_u.loc[data['date']==d]
            if data_u_d.empty:
                continue
            channel = data_u_d['Item'].values
            stas_Matrix = np.zeros((1, TotalFeature), 'int32')
            length = len(channel)
            Multi_u = min(length,Multi)# Guaranteed to have 'n'-level fingerprints
            for K in range(1, Multi_u + 1):
                ch = 0
                for KK in range(0, K):
                    ch = ch + ((channel[KK:(length - (K - KK - 1))]) * (10 ** (max_value_len * KK)))
                    # if KK==1:
                    #     ch = ch + ((channel[KK:(length - (K - KK - 1))]-ch)* (10 ** (max_value_len * KK)))
                data_u_group = pd.DataFrame(data=ch,columns=['Item'],copy=True)
                data_u_group = data_u_group.loc[data_u_group['Item'].isin(FeatureName)]
                data_u_group = data_u_group.groupby(by=['Item'])
                stas = data_u_group.size()  # Count the number of times each item appears in groups
                stas = stas.reset_index()
                stas.columns = ['Item','times']
                ChannelIndex = Inx.loc[stas['Item'].values,'inx']
                ChannelIndex = ChannelIndex.values
                stas_Matrix[0, ChannelIndex] = stas['times'] # The range of ChannelIndex obtained by each loop is different, so it will not be overwritten here, and the corresponding position is filled with the corresponding value

            Feature[J,0] = d # date
            Feature[J,1] = u # userid
            Feature[J,2:] = stas_Matrix; # feature vector of user u
            J = J+1;
    Feature = Feature[:J,:] # There is no guarantee that each user has data records for each time unit, and those without data need to be deleted

    return Feature[:,0],Feature[:,1:]

### 2021 TIFS
# ------The intersection of two user feature Spaces, The union of two user feature Spaces
def SURE(data,max_value_len,Multi = 2,ratio =1):
    # Feature extraction
    user = np.unique(data['id'])
    data['Item'] = data['Item'].astype(int)
    Feature = {} #{u:[feature vector (name) of u, feature vector (values) of u]}
    for u in user:
        data_u = data[data['id'] == u]
        channel = data_u['Item'].values
        f_u = pd.DataFrame(columns=['Item','times'])
        length = len(channel)
        Multi_u = min(length,Multi)
        for K in range(1, Multi_u+1): # Guaranteed to have 'n'-level fingerprints
            ch = 0
            for KK in range(0,K):
                ch = ch+((channel[KK:(length-(K-KK-1))])*(10**(max_value_len*KK)))
            data_u_group = pd.DataFrame(data=ch,columns=['Item'],copy=True)
            data_u_group = data_u_group.groupby(by=['Item'])
            stas = data_u_group.size()  # Count the number of times each item appears in groups
            stas = stas.reset_index()
            stas.columns = ['Item','times']
            f_u = pd.concat([f_u,stas],ignore_index=True) # concatenating feature vectors
        # Select Top
        f_u.sort_values(by='times', ascending=False, inplace=True)
        L = math.ceil(f_u.shape[0] * ratio)
        # f_u.drop(index= range(L,f_u.shape[0]),axis = 0,inplace=True)
        f_u = f_u.iloc[:L,:]
        f_u = f_u.set_index(f_u['Item']) # set index
        f_u.drop('Item',axis=1, inplace=True) # The index has saved the relevant information, delete the following part directly
        Feature[u] = f_u
    return Feature

### non overlap
def nonoverlap_ChooseFea(data,Multi=2,ratio = 0.2,max_value_len = None):
    user = np.unique(data['id'])
    data['Item'] = data['Item'].astype(int)
    if max_value_len==None:
        print('If you do not specify the number of digits of the item, problems may arise. The default is the largest number of digits in the training set.')
        # max_value_len = len(str(max(data['Item'])))
        max_value_len = np.log10(max(data['Item']))
    featureName = {}
    for u in user:
        channel = data[data['id'] == u]['Item'].values
        length = len(channel)
        Multi_u = min(length,Multi)
        feature_count = {}
        for K in range(1, Multi_u+1):
            ch = 0
            for KK in range(0,K):
                idx = [i for i in range(KK,(length//K)*K,K)] # The main difference from SURE+, here is to choose an element every K
                ch = ch+((channel[idx])*(10**(max_value_len*KK)))
            feature_count.update(dict(Counter(ch)))
        featureNum =math.ceil(len(feature_count) * ratio)
        featureName_u = dict(sorted(feature_count.items(), key=lambda x: x[1], reverse=True)[:featureNum])
        featureName.update(featureName_u)
    FeatureName = set(featureName.keys())
    return FeatureName,max_value_len
def nonoverlap_Fea(data,FeatureName,max_value_len=3,Multi=1):
    user = np.unique(data['id'])
    data['Item'] = data['Item'].astype(int)
    TotalFeature = len(FeatureName)
    Feature = np.zeros((len(user),TotalFeature),'int32')
    Label = np.zeros(len(user),'int32')

    Inx= pd.DataFrame(data=list(FeatureName),columns=['Name'],copy=True)
    Inx['inx'] = np.arange(TotalFeature)
    Inx = Inx.set_index(Inx['Name'])
    Inx.drop('Name',axis=1,inplace=True)

    J = 0;
    for u in user:
        data_u = data[data['id'] == u]
        channel = data_u['Item'].values
        stas_Matrix = np.zeros((1, TotalFeature), 'int32')
        length = len(channel)
        Multi_u = min(length,Multi)
        for K in range(1, Multi_u + 1):
            ch = 0
            for KK in range(0, K):
                idx = [i for i in range(KK,(length//K)*K,K)] # The main difference from SURE+, here is to choose an element every K
                ch = ch+((channel[idx])*(10**(max_value_len*KK)))
            data_u_group = pd.DataFrame(data=ch,columns=['Item'],copy=True)
            data_u_group = data_u_group.loc[data_u_group['Item'].isin(FeatureName)]
            data_u_group = data_u_group.groupby(by=['Item'])
            stas = data_u_group.size()
            stas = stas.reset_index()
            stas.columns = ['Item','times']
            ChannelIndex = Inx.loc[stas['Item'].values,'inx']
            ChannelIndex = ChannelIndex.values
            stas_Matrix[0, ChannelIndex] = stas['times']
        Feature[J,:] = stas_Matrix;
        Label[J] = u;
        J = J+1;
    return Feature,Label

if __name__ == '__main__':
    pass