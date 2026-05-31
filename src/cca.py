"""Canonical Correlation Analysis implementation."""

import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler


class CanonicalCorrelationAnalysis:
    """
    Canonical Correlation Analysis for finding correlations between
    two multivariate datasets.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of canonical variates to compute
    reg_param : float, default=0.0
        Regularization parameter for numerical stability
    """
    
    def __init__(self, n_components=None, reg_param=0.0):
        self.n_components = n_components
        self.reg_param = reg_param
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.loadings_X = None
        self.loadings_Y = None
        self.canonical_correlations = None
        self.canonical_variates_X = None
        self.canonical_variates_Y = None
    
    def fit(self, X, Y):
        """
        Fit CCA to two datasets.
        
        Parameters
        ----------
        X : np.ndarray
            First dataset with shape (n_samples, n_features_X)
        Y : np.ndarray
            Second dataset with shape (n_samples, n_features_Y)
        
        Returns
        -------
        self
        """
        # Standardize data
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)
        
        n_samples = X_scaled.shape[0]
        
        # Compute covariance matrices
        Cxx = (X_scaled.T @ X_scaled) / n_samples
        Cyy = (Y_scaled.T @ Y_scaled) / n_samples
        Cxy = (X_scaled.T @ Y_scaled) / n_samples
        
        # Add regularization for numerical stability
        if self.reg_param > 0:
            Cxx += self.reg_param * np.eye(Cxx.shape[0])
            Cyy += self.reg_param * np.eye(Cyy.shape[0])
        
        # Compute inverses
        try:
            Cxx_inv = np.linalg.inv(Cxx)
            Cyy_inv = np.linalg.inv(Cyy)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            Cxx_inv = np.linalg.pinv(Cxx)
            Cyy_inv = np.linalg.pinv(Cyy)
        
        # Compute canonical correlations via SVD
        M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
        U, S, _ = svd(M)
        
        # Canonical correlations
        self.canonical_correlations = np.sqrt(np.maximum(S, 0))
        
        # Compute loadings
        n_cc = min(self.n_components or min(X_scaled.shape[1], Y_scaled.shape[1]),
                   min(X_scaled.shape[1], Y_scaled.shape[1]))
        
        self.loadings_X = U[:, :n_cc]
        
        # For Y loadings
        V = Cyy_inv @ Cxy.T @ Cxx_inv @ U
        self.loadings_Y = V[:, :n_cc] / np.linalg.norm(V[:, :n_cc], axis=0, keepdims=True)
        
        return self
    
    def transform(self, X, Y):
        """
        Transform data to canonical variates.
        
        Parameters
        ----------
        X : np.ndarray
            First dataset
        Y : np.ndarray
            Second dataset
        
        Returns
        -------
        tuple
            Canonical variates for X and Y
        """
        X_scaled = self.scaler_X.transform(X)
        Y_scaled = self.scaler_Y.transform(Y)
        
        U_X = X_scaled @ self.loadings_X
        U_Y = Y_scaled @ self.loadings_Y
        
        return U_X, U_Y
    
    def fit_transform(self, X, Y):
        """
        Fit CCA and transform data.
        
        Parameters
        ----------
        X : np.ndarray
            First dataset
        Y : np.ndarray
            Second dataset
        
        Returns
        -------
        tuple
            Canonical variates for X and Y
        """
        self.fit(X, Y)
        return self.transform(X, Y)
