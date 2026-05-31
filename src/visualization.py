"""Visualization utilities for analysis results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_plotting_style():
    """
    Set up matplotlib style for publication-quality figures.
    """
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_scree_plot(pca_model, n_components=None, figsize=(10, 6), save_path=None):
    """
    Plot PCA scree plot showing explained variance.
    
    Parameters
    ----------
    pca_model : PCAAnalysis
        Fitted PCA model
    n_components : int, optional
        Number of components to display
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str or Path, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    n_comp = n_components or len(pca_model.explained_variance_ratio_)
    comps = np.arange(1, n_comp + 1)
    
    # Variance plot
    ax1.bar(comps, pca_model.explained_variance_ratio_[:n_comp])
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA Scree Plot')
    
    # Cumulative variance plot
    ax2.plot(comps, pca_model.cumulative_variance_ratio_[:n_comp], 'o-')
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_canonical_correlations(cca_model, figsize=(10, 6), save_path=None):
    """
    Plot canonical correlations.
    
    Parameters
    ----------
    cca_model : CanonicalCorrelationAnalysis
        Fitted CCA model
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str or Path, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_cc = len(cca_model.canonical_correlations)
    ax.bar(np.arange(1, n_cc + 1), cca_model.canonical_correlations)
    ax.set_xlabel('Canonical Variate')
    ax.set_ylabel('Canonical Correlation')
    ax.set_title('Canonical Correlations between SST and Precipitation')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_gmm_selection(selection_results, criterion='bic', figsize=(10, 6), save_path=None):
    """
    Plot GMM model selection results.
    
    Parameters
    ----------
    selection_results : dict
        Results from select_optimal_n_components
    criterion : str, default='bic'
        Criterion to plot
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str or Path, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_comps = sorted(selection_results.keys())
    scores = [selection_results[n][criterion] for n in n_comps]
    
    ax.plot(n_comps, scores, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel(f'{criterion.upper()} Score')
    ax.set_title(f'GMM Model Selection ({criterion.upper()})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cluster_distribution(labels, n_clusters=None, figsize=(10, 6), save_path=None):
    """
    Plot distribution of samples across clusters.
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    n_clusters : int, optional
        Number of clusters
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str or Path, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    counts = [np.sum(labels == label) for label in unique_labels]
    
    ax.bar(unique_labels, counts, color=plt.cm.Set3(unique_labels))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Distribution Across Clusters')
    
    # Add value labels on bars
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        ax.text(label, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
