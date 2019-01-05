# Mercedes-Benz-Greener-Manufacturing-Bayesian-Optimization
An implementation on tuning hyper-parameters in a regression problem (Mercedes-Benz Greener Manufacturing challenge) using Baysian Optimization (Scikit-Optimize).


Provided are a number of python scripts, each containing a pipeline composed of dimensionality reduction (pca or k-pca) and regression models ( linear, rigde regression etc). The parameters of each of these pipelines were tuned using scikit-optimize's Gaussian process minimization to obtained best possible r2 score over k-fold cross validation score. An wrapper for Bayesian optimization is also provided in `BayesianOpt.py`. 

For example, running the script `PCA_ElasticNet.py` produces the following convergence plot.
![GitHub Logo](/images/pca_elasticnet_convergence.png)

Scikit-optimize also provides functionality for plotting how objective (r2 score) varying with each hyper-parameters and also joint effect. 
![GitHub Logo](/images/pca_elasticnet_obj_func.png)


