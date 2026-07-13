"""
Script to compare distributions of ENSO events at mid-Holocene (6ka) in PC1-PC2 latent spaces.

This script compares:
1. 6ka SSTAs projected onto PC1-PC2 from piControl
2. 6ka SSTAs in their own PC1-PC2 space (no projection)

No GMM fitting is performed - only comparison of distributions in the latent spaces.
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import decomposition
import cartopy.crs as ccrs
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
PI_DATA_PATH = "/groups/XDU5/Go/research_data/latgmm/sst_pi_500.nc"
MH_DATA_PATH = "/groups/XDU5/Go/research_data/latgmm/sst_mh_500.nc"

# Output directory
OUTPUT_DIR = "./6ka_enso_comparison_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def regrid_data(ds):
    """
    Interpolate data from curvilinear grid to rectilinear grid.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with TLONG, TLAT coordinates
    
    Returns
    -------
    xr.Dataset
        Regridded dataset on 1deg x 1deg global grid
    """
    import xesmf as xe
    
    dr = ds
    
    # Rename coordinates
    ds = ds.rename({"TLONG": "lon", "TLAT": "lat"})
    
    # Create output grid (1deg x 1deg)
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 90, 1.0), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(-180, 180, 1.0), {"units": "degrees_east"}),
        }
    )
    
    # Regrid using bilinear interpolation
    regridder = xe.Regridder(ds, ds_out, "bilinear")
    dr_out = regridder(dr)
    
    return dr_out


def get_anomalies(sst_data):
    """
    Calculate monthly anomalies by removing climatological mean for each month.
    
    Parameters
    ----------
    sst_data : xr.DataArray
        SST data with time dimension
    
    Returns
    -------
    xr.DataArray
        SST anomalies
    """
    clim = sst_data.groupby('time.month').mean("time")
    anom = sst_data.groupby('time.month') - clim
    
    return anom


def reshape_for_pca(data_array):
    """
    Reshape spatial-temporal data for PCA.
    
    Parameters
    ----------
    data_array : xr.DataArray
        Data with dimensions (time, lat, lon)
    
    Returns
    -------
    np.ndarray
        Reshaped data (time, space) with NaNs removed
    """
    # Get dimensions
    n_time = data_array.shape[0]
    n_lat = data_array.shape[1]
    n_lon = data_array.shape[2]
    
    # Reshape to (time, space)
    data_2d = data_array.values.reshape(n_time, n_lat * n_lon)
    
    # Remove columns with any NaN values
    valid_mask = ~np.isnan(data_2d).any(axis=0)
    data_2d_valid = data_2d[:, valid_mask]
    
    return data_2d_valid


def perform_pca(data_2d, n_components=2):
    """
    Perform PCA on the data.
    
    Parameters
    ----------
    data_2d : np.ndarray
        Data array of shape (n_samples, n_features)
    n_components : int
        Number of PCA components to retain
    
    Returns
    -------
    tuple
        - pca object (fitted)
        - principal components scores (n_samples, n_components)
        - explained variance ratio
    """
    pca = decomposition.PCA(n_components=n_components)
    scores = pca.fit_transform(data_2d)
    
    return pca, scores, pca.explained_variance_ratio_


def project_onto_pcs(data_2d, pca_fitted):
    """
    Project data onto existing PCA components.
    
    Parameters
    ----------
    data_2d : np.ndarray
        Data array of shape (n_samples, n_features)
    pca_fitted : sklearn PCA object
        Fitted PCA model
    
    Returns
    -------
    np.ndarray
        Projected scores (n_samples, n_components)
    """
    scores = pca_fitted.transform(data_2d)
    
    return scores


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

print("=" * 80)
print("COMPARING 6KA ENSO EVENTS IN TWO PC1-PC2 LATENT SPACES")
print("=" * 80)

# 1. LOAD AND PREPROCESS DATA
print("\n[1] Loading and preprocessing data...")
ds_pi = xr.open_dataset(PI_DATA_PATH, decode_timedelta=True)
ds_mh = xr.open_dataset(MH_DATA_PATH, decode_timedelta=True)

print(f"    piControl shape: {ds_pi.sst.shape}")
print(f"    6ka (mid-Holocene) shape: {ds_mh.sst.shape}")

# Regrid both datasets
print("    Regridding to 1deg x 1deg grid...")
temp_pi_reg = regrid_data(ds_pi)
temp_mh_reg = regrid_data(ds_mh)

# Extract SST
sst_pi_500 = temp_pi_reg["sst"]
sst_mh_500 = temp_mh_reg["sst"]

# 2. CALCULATE ANOMALIES
print("\n[2] Calculating monthly anomalies...")
sst_pi_anom = get_anomalies(sst_pi_500)
sst_mh_anom = get_anomalies(sst_mh_500)

print(f"    piControl anomalies shape: {sst_pi_anom.shape}")
print(f"    6ka anomalies shape: {sst_mh_anom.shape}")

# 3. PERFORM PCA ON PICONTROL DATA
print("\n[3] Performing PCA on piControl data...")
sst_pi_2d = reshape_for_pca(sst_pi_anom)
pca_pi, scores_pi, var_pi = perform_pca(sst_pi_2d, n_components=2)

print(f"    piControl PC1 variance explained: {var_pi[0]:.4f}")
print(f"    piControl PC2 variance explained: {var_pi[1]:.4f}")
print(f"    Total variance explained (PC1+PC2): {sum(var_pi):.4f}")

# 4. OBTAIN 6KA PC SCORES - METHOD 1: PROJECTION
print("\n[4] METHOD 1: Project 6ka SSTAs onto piControl PCs...")
sst_mh_2d = reshape_for_pca(sst_mh_anom)
scores_mh_projected = project_onto_pcs(sst_mh_2d, pca_pi)

print(f"    6ka projected scores shape: {scores_mh_projected.shape}")
print(f"    PC1 mean (projected): {scores_mh_projected[:, 0].mean():.4f}")
print(f"    PC2 mean (projected): {scores_mh_projected[:, 1].mean():.4f}")

# 5. OBTAIN 6KA PC SCORES - METHOD 2: OWN PCA
print("\n[5] METHOD 2: Perform PCA on 6ka SSTAs independently...")
pca_mh, scores_mh_own, var_mh = perform_pca(sst_mh_2d, n_components=2)

print(f"    6ka PC1 variance explained: {var_mh[0]:.4f}")
print(f"    6ka PC2 variance explained: {var_mh[1]:.4f}")
print(f"    Total variance explained (PC1+PC2): {sum(var_mh):.4f}")
print(f"    PC1 mean (own PCA): {scores_mh_own[:, 0].mean():.4f}")
print(f"    PC2 mean (own PCA): {scores_mh_own[:, 1].mean():.4f}")

# 6. STATISTICAL COMPARISON
print("\n[6] Statistical Comparison of Distributions...")

# PC1 comparisons
pc1_proj = scores_mh_projected[:, 0]
pc1_own = scores_mh_own[:, 0]

ks_stat_pc1, ks_pval_pc1 = stats.ks_2samp(pc1_proj, pc1_own)
print(f"\n    PC1 Kolmogorov-Smirnov test:")
print(f"      Test statistic: {ks_stat_pc1:.6f}")
print(f"      p-value: {ks_pval_pc1:.6e}")

# PC2 comparisons
pc2_proj = scores_mh_projected[:, 1]
pc2_own = scores_mh_own[:, 1]

ks_stat_pc2, ks_pval_pc2 = stats.ks_2samp(pc2_proj, pc2_own)
print(f"\n    PC2 Kolmogorov-Smirnov test:")
print(f"      Test statistic: {ks_stat_pc2:.6f}")
print(f"      p-value: {ks_pval_pc2:.6e}")

# Descriptive statistics
print(f"\n    PC1 Statistics:")
print(f"      Projected - Mean: {pc1_proj.mean():.4f}, Std: {pc1_proj.std():.4f}")
print(f"      Own PCA  - Mean: {pc1_own.mean():.4f}, Std: {pc1_own.std():.4f}")

print(f"\n    PC2 Statistics:")
print(f"      Projected - Mean: {pc2_proj.mean():.4f}, Std: {pc2_proj.std():.4f}")
print(f"      Own PCA  - Mean: {pc2_own.mean():.4f}, Std: {pc2_own.std():.4f}")

# 7. VISUALIZATION
print("\n[7] Creating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Scatter plot - Projected
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(pc1_proj, pc2_proj, alpha=0.5, s=20, c='blue')
ax1.set_xlabel(f'PC1 ({var_pi[0]:.2%} variance)')
ax1.set_ylabel(f'PC2 ({var_pi[1]:.2%} variance)')
ax1.set_title('6ka SSTAs Projected onto\npiControl PC1-PC2')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

# Plot 2: Scatter plot - Own PCA
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(pc1_own, pc2_own, alpha=0.5, s=20, c='red')
ax2.set_xlabel(f'PC1 ({var_mh[0]:.2%} variance)')
ax2.set_ylabel(f'PC2 ({var_mh[1]:.2%} variance)')
ax2.set_title('6ka SSTAs in Own\nPC1-PC2 Space')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax2.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

# Plot 3: Overlay comparison
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(pc1_proj, pc2_proj, alpha=0.4, s=20, c='blue', label='Projected')
ax3.scatter(pc1_own, pc2_own, alpha=0.4, s=20, c='red', label='Own PCA')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_title('Overlay: Projected vs Own PCA')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax3.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

# Plot 4: PC1 distribution - Histogram
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(pc1_proj, bins=50, alpha=0.6, label='Projected', color='blue', density=True)
ax4.hist(pc1_own, bins=50, alpha=0.6, label='Own PCA', color='red', density=True)
ax4.set_xlabel('PC1')
ax4.set_ylabel('Density')
ax4.set_title(f'PC1 Distribution\n(KS test p-value: {ks_pval_pc1:.2e})')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: PC2 distribution - Histogram
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(pc2_proj, bins=50, alpha=0.6, label='Projected', color='blue', density=True)
ax5.hist(pc2_own, bins=50, alpha=0.6, label='Own PCA', color='red', density=True)
ax5.set_xlabel('PC2')
ax5.set_ylabel('Density')
ax5.set_title(f'PC2 Distribution\n(KS test p-value: {ks_pval_pc2:.2e})')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Q-Q plot for PC1
ax6 = fig.add_subplot(gs[1, 2])
stats.probplot(pc1_proj, dist="norm", plot=ax6)
ax6.set_title('Q-Q Plot: PC1 (Projected)')
ax6.grid(True, alpha=0.3)

# Plot 7: Q-Q plot for PC2
ax7 = fig.add_subplot(gs[2, 0])
stats.probplot(pc2_proj, dist="norm", plot=ax7)
ax7.set_title('Q-Q Plot: PC2 (Projected)')
ax7.grid(True, alpha=0.3)

# Plot 8: Density plot PC1
ax8 = fig.add_subplot(gs[2, 1])
pc1_proj_sorted = np.sort(pc1_proj)
pc1_own_sorted = np.sort(pc1_own)
from scipy.stats import gaussian_kde
kde_proj = gaussian_kde(pc1_proj)
kde_own = gaussian_kde(pc1_own)
x_range = np.linspace(min(pc1_proj.min(), pc1_own.min()), 
                      max(pc1_proj.max(), pc1_own.max()), 200)
ax8.plot(x_range, kde_proj(x_range), 'b-', label='Projected', linewidth=2)
ax8.plot(x_range, kde_own(x_range), 'r-', label='Own PCA', linewidth=2)
ax8.fill_between(x_range, kde_proj(x_range), alpha=0.3, color='blue')
ax8.fill_between(x_range, kde_own(x_range), alpha=0.3, color='red')
ax8.set_xlabel('PC1')
ax8.set_ylabel('Density')
ax8.set_title('PC1 KDE Comparison')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Density plot PC2
ax9 = fig.add_subplot(gs[2, 2])
kde_proj_pc2 = gaussian_kde(pc2_proj)
kde_own_pc2 = gaussian_kde(pc2_own)
x_range_pc2 = np.linspace(min(pc2_proj.min(), pc2_own.min()), 
                          max(pc2_proj.max(), pc2_own.max()), 200)
ax9.plot(x_range_pc2, kde_proj_pc2(x_range_pc2), 'b-', label='Projected', linewidth=2)
ax9.plot(x_range_pc2, kde_own_pc2(x_range_pc2), 'r-', label='Own PCA', linewidth=2)
ax9.fill_between(x_range_pc2, kde_proj_pc2(x_range_pc2), alpha=0.3, color='blue')
ax9.fill_between(x_range_pc2, kde_own_pc2(x_range_pc2), alpha=0.3, color='red')
ax9.set_xlabel('PC2')
ax9.set_ylabel('Density')
ax9.set_title('PC2 KDE Comparison')
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

plt.savefig(os.path.join(OUTPUT_DIR, '6ka_enso_pc_comparison.png'), dpi=300, bbox_inches='tight')
print(f"    Saved: {os.path.join(OUTPUT_DIR, '6ka_enso_pc_comparison.png')}")
plt.close()

# 8. SAVE RESULTS TO CSV
print("\n[8] Saving results...")

# Create summary dataframe
results_df = pd.DataFrame({
    'Metric': [
        'PC1_projected_mean', 'PC1_projected_std', 'PC1_projected_min', 'PC1_projected_max',
        'PC1_own_mean', 'PC1_own_std', 'PC1_own_min', 'PC1_own_max',
        'PC2_projected_mean', 'PC2_projected_std', 'PC2_projected_min', 'PC2_projected_max',
        'PC2_own_mean', 'PC2_own_std', 'PC2_own_min', 'PC2_own_max',
        'PC1_KS_statistic', 'PC1_KS_pvalue',
        'PC2_KS_statistic', 'PC2_KS_pvalue',
        'piControl_PC1_variance', 'piControl_PC2_variance',
        '6ka_PC1_variance', '6ka_PC2_variance'
    ],
    'Value': [
        pc1_proj.mean(), pc1_proj.std(), pc1_proj.min(), pc1_proj.max(),
        pc1_own.mean(), pc1_own.std(), pc1_own.min(), pc1_own.max(),
        pc2_proj.mean(), pc2_proj.std(), pc2_proj.min(), pc2_proj.max(),
        pc2_own.mean(), pc2_own.std(), pc2_own.min(), pc2_own.max(),
        ks_stat_pc1, ks_pval_pc1,
        ks_stat_pc2, ks_pval_pc2,
        var_pi[0], var_pi[1],
        var_mh[0], var_mh[1]
    ]
})

results_df.to_csv(os.path.join(OUTPUT_DIR, 'comparison_statistics.csv'), index=False)
print(f"    Saved: {os.path.join(OUTPUT_DIR, 'comparison_statistics.csv')}")

# Save PC scores
scores_df = pd.DataFrame({
    'PC1_projected': pc1_proj,
    'PC2_projected': pc2_proj,
    'PC1_own': pc1_own,
    'PC2_own': pc2_own
})
scores_df.to_csv(os.path.join(OUTPUT_DIR, '6ka_pc_scores.csv'), index=False)
print(f"    Saved: {os.path.join(OUTPUT_DIR, '6ka_pc_scores.csv')}")

# 9. SUMMARY
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"\nVariance Explained:")
print(f"  piControl PC1: {var_pi[0]:.4f}")
print(f"  piControl PC2: {var_pi[1]:.4f}")
print(f"  6ka PC1 (own): {var_mh[0]:.4f}")
print(f"  6ka PC2 (own): {var_mh[1]:.4f}")

print(f"\nDistribution Differences (KS test):")
print(f"  PC1: statistic={ks_stat_pc1:.6f}, p-value={ks_pval_pc1:.2e}")
print(f"  PC2: statistic={ks_stat_pc2:.6f}, p-value={ks_pval_pc2:.2e}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("=" * 80)
