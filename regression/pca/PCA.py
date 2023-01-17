import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
    
    def fit(self, X, std=True, replace_X = False):
        n = X.shape[0]
        m = X.shape[1]

        if replace_X != False:
            X = X.copy()

        self.mean = np.mean(X, axis = 0)
        self.scale = np.std(X, axis = 0)
        X_std = (X - self.mean) / self.scale


        covar = (X_std.T @ X_std)/(n-1) #calculation of covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(covar) #eigendecompsition of covariance matrix
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs *= signs[np.newaxis, :]
        eig_vecs = eig_vecs.T

        print(eig_vecs.shape)

        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i, :]) for i in range(len(eig_vals))]
        print(np.array(eig_pairs).shape)
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs]) 

        self.components = eig_vecs_sorted[:self.n_components, :]

        self.explained_variable_ratio = [i/np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]
        self.cum_explained_variance = np.cumsum(self.explained_variable_ratio)

        return {
            "Components": self.components,
            "Explained Variable Ratio": self.explained_variable_ratio, 
            "Cumulative Explained Variance": self.cum_explained_variance
        }

    def transform(self, X, replace_X = False):
        if replace_X != False:
            X = X.copy()

        X_std = (X - self.mean) / self.scale
        X_proj = X_std.dot(self.components.T)
        
        return X_proj