"""Gaussian Mixture Model clustering implementation."""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


class GaussianMixtureModel:
    """
    Gaussian Mixture Model for clustering with model selection.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of mixture components
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        Type of covariance parameters
    init_params : {'kmeans', 'random'}, default='kmeans'
        Method for initialization
    max_iter : int, default=100
        Maximum number of EM iterations
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_components=2, covariance_type='full',
                 init_params='kmeans', max_iter=100, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.max_iter = max_iter
        self.random_state = random_state
        self.gmm = None
        self.labels = None
        self.probabilities = None
    
    def fit(self, X):
        """
        Fit GMM to data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data with shape (n_samples, n_features)
        
        Returns
        -------
        self
        """
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            init_params=self.init_params,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.gmm.fit(X)
        self.labels = self.gmm.predict(X)
        self.probabilities = self.gmm.predict_proba(X)
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        
        Returns
        -------
        np.ndarray
            Cluster labels
        """
        return self.gmm.predict(X)
    
    def predict_proba(self, X):
        """
        Predict cluster probabilities for new data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        
        Returns
        -------
        np.ndarray
            Cluster probabilities
        """
        return self.gmm.predict_proba(X)
    
    @property
    def means_(self):
        """Cluster means."""
        return self.gmm.means_
    
    @property
    def covariances_(self):
        """Cluster covariances."""
        return self.gmm.covariances_
    
    @property
    def weights_(self):
        """Mixing coefficients (weights)."""
        return self.gmm.weights_
    
    @property
    def bic_(self):
        """Bayesian Information Criterion."""
        return self.gmm.bic(self.gmm.means_)
    
    @property
    def aic_(self):
        """Akaike Information Criterion."""
        return self.gmm.aic(self.gmm.means_)
    
    def compute_metrics(self, X, metric='silhouette'):
        """
        Compute clustering quality metrics.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        metric : {'silhouette', 'davies_bouldin'}
            Metric to compute
        
        Returns
        -------
        float
            Metric value
        """
        labels = self.predict(X)
        
        if metric == 'silhouette':
            if len(np.unique(labels)) > 1:
                return silhouette_score(X, labels)
            return -1
        
        elif metric == 'davies_bouldin':
            if len(np.unique(labels)) > 1:
                return davies_bouldin_score(X, labels)
            return float('inf')


def select_optimal_n_components(X, n_components_range, criterion='bic'):
    """
    Select optimal number of components using information criteria.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    n_components_range : range or list
        Range of component numbers to test
    criterion : {'bic', 'aic', 'silhouette'}
        Selection criterion
    
    Returns
    -------
    dict
        Dictionary with results for each n_components
    """
    results = {}
    
    for n_comp in n_components_range:
        gmm = GaussianMixtureModel(n_components=n_comp)
        gmm.fit(X)
        
        if criterion == 'bic':
            score = gmm.bic_
        elif criterion == 'aic':
            score = gmm.aic_
        elif criterion == 'silhouette':
            score = gmm.compute_metrics(X, metric='silhouette')
        
        results[n_comp] = {
            'score': score,
            'bic': gmm.bic_,
            'aic': gmm.aic_,
            'model': gmm
        }
    
    return results
