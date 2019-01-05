from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from utils import *
from BayesianOpt import *
from skopt.space import Integer, Real
from sklearn.metrics import r2_score


# objective function with r2 cross validation score to be optimized
def PCA_Regressor(n_components):
    n_components = int(n_components[0])
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X_Train);
    X_PCA = pca.transform(X_Train)
    LinearReg = LinearRegression()
    mean_r2 =  -np.mean(cross_val_score(LinearReg, X_PCA, Y_Train_scaled, cv=5, scoring='r2'))
    return mean_r2
    
# function to run bayesian optimization 
def RunBayesianParameterSearch(n_calls=50):
    space = [Integer(10,500, name='n_components')]
    PCA_LinearReg_Optimizer = BayesianOptimizer(PCA_Regressor, space)
    PCA_LinearReg_Optimizer.Optimize(n_calls=n_calls)
    PCA_LinearReg_Optimizer.PlotConvergence()
    PCA_LinearReg_Optimizer.PlotObjective()
    PCA_LinearReg_Optimizer.PlotEvaluations()
    optimal_n_components = PCA_LinearReg_Optimizer.OptimalParameters()
    SavePickle('./BayesianOptimizationResults/LinearReg_PCA_BayesianOptimization.pickle',PCA_LinearReg_Optimizer)
    
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
        RunBayesianParameterSearch(n_calls=20)    
    PCA_LinearReg_Optimizer = ReadPickle('./BayesianOptimizationResults/LinearReg_PCA_BayesianOptimization.pickle')
    Optimal_n_components = PCA_LinearReg_Optimizer.OptimalParameters()[0]

    # re-training pipeline with optimal parameters on complete training set
    pca = PCA(n_components = Optimal_n_components)
    pca.fit(X_Train)
    X_PCA = pca.transform(X_Train); X_PCA_Val = pca.transform(X_Val)
    linearReg = LinearRegression()
    linearReg.fit(X_PCA, Y_Train_scaled)
    Y_Val_Pred = linearReg.predict(X_PCA_Val)
    print('r2 Score on Training Set    = ', r2_score(Y_Train_scaled, linearReg.predict(X_PCA)))
    print('r2 Score on scaled Y values = ', r2_score(Y_Val_scaled, Y_Val_Pred))
    Y_Val_Pred = scalerY.inverse_transform(Y_Val_Pred)
    print('r2 Score on orignl Y values = ', r2_score(Y_Val, Y_Val_Pred))
    
    
