from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import KernelPCA
from utils import *
from BayesianOpt import *
from skopt.space import Integer,Categorical, Real

# objective function with r2 cross validation score to be optimized
def KernelPCA_Regressor(param):
    print(param)
    n_components, gamma = param
    kernelpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma, degree=1, n_jobs=1)
    kernelpca.fit(X_Train);
    X_KPCA = kernelpca.transform(X_Train)
    LinearReg = LinearRegression()
    mean_r2 =  -np.mean(cross_val_score(LinearReg, X_KPCA, Y_Train_scaled, cv=5, scoring='r2'))
    return mean_r2

# function to run bayesian optimization 
def RunBayesianParameterSearch(n_calls=50):
    space = [
             Integer(10,500, name='n_components'), 
             Real(0.00001,1.0, name='Gamma'),
             ]
    KernelPCA_LinearReg_Optimizer = BayesianOptimizer(KernelPCA_Regressor, space)
    KernelPCA_LinearReg_Optimizer.Optimize(n_calls=n_calls)
    optimal_params = KernelPCA_LinearReg_Optimizer.OptimalParameters()
    SavePickle('./BayesianOptimizationResults/LinearReg_KernelPCA_BayesianOptimization.pickle', 
               KernelPCA_LinearReg_Optimizer)

    
###############################################################################
RUN_BAYESIAN_PARAMETER_SEARCH = False

if __name__ == '__main__':
    
    # loading dataset and removing outliers in target values that are > ylimit
    dataloader = DataLoader('./Data/train.csv', './Data/test.csv')
    dataloader.Load_csv()
    ylimit = [180]
    dataloader.Set_UpperLimit(ylimit)
    X_Train, Y_Train, _ = dataloader.Get_Data(one_hot_encoded = True)
    
    # splitting training data into train and validation set
    X_Train, X_Val, Y_Train, Y_Val = Train_Test_Split(X_Train, Y_Train, ratio=0.33, shuffle=False)
    
    
    # pre-processing dataset
    scalerY = StandardScaler(); scalerY.fit(Y_Train[:,np.newaxis]);
    Y_Train_scaled = scalerY.transform(Y_Train[:,np.newaxis])[:,0]
    Y_Val_scaled = scalerY.transform(Y_Val[:,np.newaxis])[:,0]
    
    scalerX = StandardScaler(); scalerX.fit(X_Train);
    X_Train = scalerX.transform(X_Train)
    X_Val = scalerX.transform(X_Val)
    
    # either re-run bayesian optimization or load previous saved pickle file
    if RUN_BAYESIAN_PARAMETER_SEARCH:
        RunBayesianParameterSearch(n_calls=100)
    KernelPCA_LinearReg_Optimizer = ReadPickle('./BayesianOptimizationResults/LinearReg_KernelPCA_BayesianOptimization.pickle')
    optimal_params = KernelPCA_LinearReg_Optimizer.OptimalParameters()
    
    # re-training pipeline with optimal parameters on complete training set
    kernelpca = KernelPCA(n_components = optimal_params[0], kernel = 'rbf', gamma = optimal_params[1])
    kernelpca.fit(X_Train)
    X_KPCA = kernelpca.transform(X_Train); X_KPCA_Val = kernelpca.transform(X_Val)
    linearReg = LinearRegression()
    linearReg.fit(X_KPCA, Y_Train_scaled)
    Y_Val_Pred = linearReg.predict(X_KPCA_Val)
    print('r2 Score on Training Set    = ', r2_score(Y_Train_scaled, linearReg.predict(X_KPCA)))
    print('r2 Score on scaled Y values = ', r2_score(Y_Val_scaled, Y_Val_Pred))
    Y_Val_Pred = scalerY.inverse_transform(Y_Val_Pred)
    print('r2 Score on orignl Y values = ', r2_score(Y_Val, Y_Val_Pred))
    
    