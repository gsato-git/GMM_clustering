"""Utility functions."""

import numpy as np
import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to configuration file
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    config_path : str or Path
        Path to save configuration
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_output_directories(output_dirs):
    """
    Create output directories.
    
    Parameters
    ----------
    output_dirs : list or dict
        List of directory paths or dict with path:description
    """
    if isinstance(output_dirs, dict):
        output_dirs = output_dirs.keys()
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def compute_correlation_matrix(X, Y=None):
    """
    Compute correlation matrix.
    
    Parameters
    ----------
    X : np.ndarray
        First dataset
    Y : np.ndarray, optional
        Second dataset. If None, compute auto-correlation of X
    
    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    if Y is None:
        Y = X
    
    # Standardize
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Y_std = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    
    n = X_std.shape[0]
    corr = (X_std.T @ Y_std) / (n - 1)
    
    return corr


def compute_statistics(X, labels):
    """
    Compute statistics for each cluster.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    labels : np.ndarray
        Cluster labels
    
    Returns
    -------
    dict
        Dictionary with statistics for each cluster
    """
    stats = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        cluster_data = X[mask]
        
        stats[label] = {
            'n_samples': np.sum(mask),
            'mean': np.mean(cluster_data, axis=0),
            'std': np.std(cluster_data, axis=0),
            'min': np.min(cluster_data, axis=0),
            'max': np.max(cluster_data, axis=0)
        }
    
    return stats


def print_summary_statistics(stats):
    """
    Print summary statistics.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_statistics
    """
    for cluster_id, cluster_stats in stats.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Samples: {cluster_stats['n_samples']}")
        print(f"  Mean: {cluster_stats['mean']}")
        print(f"  Std: {cluster_stats['std']}")
