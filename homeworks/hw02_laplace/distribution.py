import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        if(x.ndim > 1):
            return np.mean(np.abs(x - np.median(x, axis = 0)), axis = 0)
        else:
            return np.mean(np.abs(x - np.median(x)))
        ####

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        if(features.ndim > 1):
            self.loc = np.median(features, axis = 0)
        else:
            self.loc = np.median(features)
        self.scale = self.mean_abs_deviation_from_median(features)
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        if(values.ndim > 1):
            return np.array([np.log(np.exp(-1 * np.abs(values[i] - self.loc) / self.scale) / (2*self.scale)) for i in range(values.shape[0])])
        else:
            return np.log(np.exp(-1 * np.abs(values - self.loc) / self.scale) / (2*self.scale))
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
