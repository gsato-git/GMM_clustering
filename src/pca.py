"""Principal Component Analysis implementation."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAnalysis:
    """
    Principal Component Analysis for dimensionality reduction.
    
    Parameters
    ----------
    n_components : int or float, default=None
        Number of components to keep. If float between 0 and 1,
        select components to explain that proportion of variance.
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.data_mean = None
        self.data_std = None
    
    def fit(self, X):
        """
        Fit PCA to data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data with shape (n_samples, n_features)
        
        Returns
        -------
        self
        """
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)
        self.data_mean = self.scaler.mean_
        self.data_std = self.scaler.scale_
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        return self
    
    def transform(self, X):
        """
        Transform data to principal components.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        
        Returns
        -------
        np.ndarray
            Principal components with shape (n_samples, n_components)
        """
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        
        Returns
        -------
        np.ndarray
            Principal components
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_pca):
        """
        Transform principal components back to original space.
        
        Parameters
        ----------
        X_pca : np.ndarray
            Principal components
        
        Returns
        -------
        np.ndarray
            Reconstructed data in original space
        """
        X_scaled = self.pca.inverse_transform(X_pca)
        return self.scaler.inverse_transform(X_scaled)
    
    @property
    def explained_variance_ratio_(self):
        """Proportion of variance explained by each component."""
        return self.pca.explained_variance_ratio_
    
    @property
    def cumulative_variance_ratio_(self):
        """Cumulative proportion of variance explained."""
        return np.cumsum(self.pca.explained_variance_ratio_)
    
    @property
    def components_(self):
        """Principal component loadings (eigenvectors)."""
        return self.pca.components_
    
    @property
    def explained_variance_(self):
        """Variance of each principal component."""
        return self.pca.explained_variance_
    
    def get_loadings(self):
        """
        Get PCA loadings (components scaled by singular values).
        
        Returns
        -------
        np.ndarray
            Loadings with shape (n_features, n_components)
        """
        return self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
    
    def scree_plot_data(self):
        """
        Get data for scree plot.
        
        Returns
        -------
        dict
            Dictionary with variance and cumulative variance data
        """
        return {
            'variance': self.explained_variance_ratio_,
            'cumulative_variance': self.cumulative_variance_ratio_,
            'n_components': len(self.explained_variance_ratio_)
        }
