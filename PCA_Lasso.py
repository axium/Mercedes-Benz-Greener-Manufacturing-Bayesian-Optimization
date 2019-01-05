from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from utils import *
from BayesianOpt import *
from skopt.space import Integer, Real
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline



# pipeLine to be optimized 
def CreatePipeline(n_components = None, alpha = None, ):
    pca = PCA(n_components = n_components)
    lasso = Lasso(alpha = alpha)
    pca_lasso = Pipeline([('pca', pca), ('lasso', lasso)])
    return pca_lasso

# objective function with r2 cross validation score to be optimized
def PCA_Lasso_Regressor(params):
    n_components, alpha = params
    pca_lasso = CreatePipeline(n_components = n_components, alpha = alpha)
    pca_lasso.fit(X_Train, Y_Train_scaled);
    mean_r2 =  -np.mean(cross_val_score(pca_lasso, X_Train, Y_Train_scaled, cv=5, scoring='r2'))
    return mean_r2
    
# function to run bayesian optimization 
def RunBayesianParameterSearch(n_calls=50):
    space = [
             Integer(10,X_Train.shape[1], name='n_components'),
             Real(0.0001, 10, name='alpha')
             ]
    PCA_Lasso_Optimizer = BayesianOptimizer(PCA_Lasso_Regressor, space)
    PCA_Lasso_Optimizer.Optimize(n_calls=n_calls)
    PCA_Lasso_Optimizer.PlotConvergence()
    PCA_Lasso_Optimizer.PlotObjective()
    PCA_Lasso_Optimizer.PlotEvaluations()
    optimal_n_components = PCA_Lasso_Optimizer.OptimalParameters()
    SavePickle('./BayesianOptimizationResults/Lasso_PCA_BayesianOptimization.pickle',PCA_Lasso_Optimizer)
    
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
    PCA_Lasso_Optimizer = ReadPickle('./BayesianOptimizationResults/Lasso_PCA_BayesianOptimization.pickle')
    optimal_params = PCA_Lasso_Optimizer.OptimalParameters()
    
    # re-training pipeline with optimal parameters on complete training set
    pca_lasso = CreatePipeline(n_components = optimal_params[0], alpha = optimal_params[1])
    pca_lasso.fit(X_Train, Y_Train_scaled)
    Y_Val_Pred = pca_lasso.predict(X_Val)
    print('r2 Score on Training Set    = ', r2_score(Y_Train_scaled, pca_lasso.predict(X_Train)))
    print('r2 Score on scaled Y values = ', r2_score(Y_Val_scaled, Y_Val_Pred))
    Y_Val_Pred = scalerY.inverse_transform(Y_Val_Pred)
    print('r2 Score on orignl Y values = ', r2_score(Y_Val, Y_Val_Pred))
    
    