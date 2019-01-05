import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
    
    
class BayesianOptimizer():
    '''
    A wrapper class for Bayesian Optimization using scikit-optimize. Requires
    a python function which is to be optimized with search space. Provides
    abstract methods to start optimizing, return optimal parameters and plot
    methods. It can be saved as a pickle object to be used later.
    
    '''
    
    def __init__(self, func,  space):
        self.func = func
        self.space = space
        self.results = []
        
    def Optimize(self, n_calls, return_results = True):
        self.results = gp_minimize(self.func, self.space, n_calls=n_calls, verbose=True)
        if return_results:
            return self.results
    def MinFuncValue(self):
        return self.results.fun
    
    def OptimalParameters(self):
        return self.results.x
    
    def PlotConvergence(self):
        plot_convergence(self.results)
        
    def PlotObjective(self):
        plot_objective(self.results)
        
    def PlotEvaluations(self):
        plot_evaluations(self.results)
