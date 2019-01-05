import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle



class DataLoader():
    '''
    This is DataLoader class to handle csv loading and converting categorical
    features to one-hot-encoded features.
    '''
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.df_Train = []
        self.df_Test = []
        
    def Load_csv(self):
        self.df_Train = pd.read_csv(self.train_path)
        self.df_Test = pd.read_csv(self.test_path)


    def Get_Data(self, one_hot_encoded = False):
        Y_Train = self.df_Train['y']
        X_Train = self.df_Train.drop(['y', 'ID'] , 1)
        X_Test  = self.df_Test.drop('ID' , 1)
        
        if one_hot_encoded == False:
            return X_Train, Y_Train, X_Test
        else:
            X_concat = pd.concat([ X_Train , X_Test])
            X_concat_Numeric = pd.get_dummies(X_concat, columns=['X0','X1','X2','X3','X4','X5','X6','X8'])
            X_concat_Numeric = X_concat_Numeric.astype('int64')
            
            X_Train = X_concat_Numeric[:self.df_Train.shape[0]]
            X_Test  = X_concat_Numeric[self.df_Train.shape[0]:]
            return X_Train, Y_Train, X_Test
        
    def Get_DFs(self):
        return self.df_Train, self.df_Test

    def Set_UpperLimit(self, UppLim):
        for upplim in UppLim:
            self.df_Train['y'].loc[ self.df_Train['y'] > upplim] = upplim
            
            

def Train_Test_Split(X, Y, ratio, shuffle=True):
    '''
    Shuffles the data and splits train_validation data
    '''
    if shuffle:
        for i in range(10):
            idx = np.random.permutation(X.shape[0])
            X = X[idx] ; Y = Y[idx]
    X_Train, X_Val, Y_Train, Y_Val = train_test_split( X, Y, test_size=ratio)
    return X_Train, X_Val, Y_Train, Y_Val



def SavePickle(path, obj):
    pickle.dump( obj, open( path, "wb" ) )
    
def ReadPickle(path):
    return pickle.load( open( path, "rb" ) )
    