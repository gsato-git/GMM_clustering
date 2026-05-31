# Gaussian Mixture Model (GMM) Clustering with Climate Data

A comprehensive analysis framework for clustering climate data using Principal Component Analysis (PCA), Canonical Correlation Analysis (CCA), and Gaussian Mixture Models (GMM). This project focuses on Sea Surface Temperature (SST) and precipitation data analysis.

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
  - [Canonical Correlation Analysis (CCA)](#canonical-correlation-analysis-cca)
  - [Gaussian Mixture Model (GMM)](#gaussian-mixture-model-gmm)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository provides tools and scripts for analyzing climate datasets using a three-stage analytical pipeline:

1. **PCA**: Dimensionality reduction of high-dimensional climate data
2. **CCA**: Identifying correlations between Sea Surface Temperature and precipitation patterns
3. **GMM**: Clustering data into distinct climate patterns and regimes

The analysis enables identification of coherent climate patterns and their relationships across different regions and time scales.

## Methodology

### Principal Component Analysis (PCA)

**Purpose**: Reduce dimensionality while preserving maximum variance in the data.

**Application**: 
- Extract leading modes of variability from SST and precipitation fields
- Identify dominant spatial patterns and temporal evolution
- Reduce computational complexity for downstream analyses

**Key Benefits**:
- Handles multicollinearity in climate variables
- Provides orthogonal components for interpretation
- Enables visualization of high-dimensional climate data

**Mathematical Foundation**:
```
X = U * Σ * V^T
```
Where X is the standardized data matrix, U contains principal components, Σ contains singular values, and V^T contains loadings.

### Canonical Correlation Analysis (CCA)

**Purpose**: Discover relationships between two multivariate datasets (SST and precipitation).

**Application**:
- Identify canonical variates that maximize correlation between SST and precipitation
- Understand how oceanic and atmospheric patterns co-vary
- Extract coupled climate modes (e.g., ENSO-like teleconnections)

**Key Benefits**:
- Reveals hidden correlations between different climate variables
- Provides interpretable canonical variates
- Useful for predictability analysis

**Mathematical Foundation**:
- Maximizes correlation between linear combinations of two datasets
- Produces pairs of canonical variates with maximum canonical correlation
- Useful for understanding SST-precipitation teleconnections

### Gaussian Mixture Model (GMM)

**Purpose**: Identify distinct climate regimes through probabilistic clustering.

**Application**:
- Partition climate data into K distinct clusters/regimes
- Assign probability of each observation belonging to each cluster
- Characterize dominant climate patterns and their persistence

**Key Benefits**:
- Soft clustering provides probabilistic interpretation
- Can identify transitions between climate states
- BIC/AIC criteria for optimal cluster selection
- Interpretable cluster statistics (means, covariances)

**Mathematical Foundation**:
```
p(x) = Σ π_k * N(x | μ_k, Σ_k)
```
Where:
- π_k: mixing coefficients (cluster probabilities)
- N: multivariate Gaussian distribution
- μ_k: cluster means
- Σ_k: cluster covariances

## Dataset

### Sea Surface Temperature (SST) Data
- **Source**: NOAA/OISST or similar gridded products
- **Variables**: Monthly mean sea surface temperature anomalies
- **Domain**: Global or regional (customizable)
- **Temporal Coverage**: Multiple decades for robust statistics
- **Preprocessing**: Climatology removal, standardization, quality control

### Precipitation Data
- **Source**: GPCC, CMAP, or similar precipitation products
- **Variables**: Monthly total precipitation or anomalies
- **Domain**: Land regions (customizable)
- **Temporal Coverage**: Aligned with SST data
- **Preprocessing**: Logarithmic transformation (optional), standardization

## Project Structure

```
GMM_clustering/
├── README.md                          # This file
├── LICENSE                            # Project license
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── config/
│   └── config.yaml                   # Configuration parameters
├── data/
│   ├── raw/                          # Raw data files
│   ├── processed/                    # Processed/preprocessed data
│   └── README.md                     # Data documentation
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Initial data analysis
│   ├── 02_pca_analysis.ipynb        # PCA implementation and results
│   ├── 03_cca_analysis.ipynb        # CCA implementation and results
│   └── 04_gmm_clustering.ipynb      # GMM clustering and validation
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocessing.py             # Data preprocessing functions
│   ├── pca.py                       # PCA implementation
│   ├── cca.py                       # CCA implementation
│   ├── gmm.py                       # GMM clustering
│   ├── visualization.py             # Plotting utilities
│   └── utils.py                     # Utility functions
├── results/
│   ├── figures/                     # Generated plots
│   ├── models/                      # Trained models
│   └── reports/                     # Analysis reports
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_pca.py
    ├── test_cca.py
    └── test_gmm.py
```

## Installation

### Prerequisites
- Python 3.8+
- conda or pip

### Step 1: Clone the Repository
```bash
git clone https://github.com/gsato-git/GMM_clustering.git
cd GMM_clustering
```

### Step 2: Create Virtual Environment
```bash
# Using conda
conda create -n gmm-clustering python=3.9
conda activate gmm-clustering

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Basic Workflow

#### 1. Data Preparation
```python
from src.data_loader import load_sst_data, load_precipitation_data
from src.preprocessing import standardize, remove_climatology

# Load data
sst = load_sst_data('data/raw/sst.nc')
precip = load_precipitation_data('data/raw/precip.nc')

# Preprocess
sst_anom = remove_climatology(sst)
sst_std = standardize(sst_anom)
```

#### 2. PCA Analysis
```python
from src.pca import PCAAnalysis

pca = PCAAnalysis(n_components=10)
pca.fit(sst_std)
pcs = pca.transform(sst_std)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

#### 3. CCA Analysis
```python
from src.cca import CanonicalCorrelationAnalysis

precip_std = standardize(precip)

cca = CanonicalCorrelationAnalysis(n_components=5)
cca.fit(sst_std, precip_std)
print(f"Canonical correlations: {cca.canonical_correlations}")
```

#### 4. GMM Clustering
```python
from src.gmm import GaussianMixtureModel

# Determine optimal number of clusters
gmm_results = {}
for n_clusters in range(2, 8):
    gmm = GaussianMixtureModel(n_components=n_clusters)
    gmm.fit(pcs)  # Use PCA-reduced data
    gmm_results[n_clusters] = {'bic': gmm.bic_, 'aic': gmm.aic_}

# Fit final model
optimal_k = min(gmm_results, key=lambda x: gmm_results[x]['bic'])
gmm_final = GaussianMixtureModel(n_components=optimal_k)
gmm_final.fit(pcs)
labels = gmm_final.predict(pcs)
```

### Running Notebooks

Execute the analysis pipeline through Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

Follow notebooks in order:
1. `01_data_exploration.ipynb` - Understand your data
2. `02_pca_analysis.ipynb` - Extract principal components
3. `03_cca_analysis.ipynb` - Analyze SST-precipitation relationships
4. `04_gmm_clustering.ipynb` - Identify climate regimes

## Results

The analysis generates:

### Figures
- PCA scree plots and principal components
- Canonical correlation analysis results
- GMM cluster assignments and transition probabilities
- Spatial patterns of identified climate regimes
- Time series of cluster memberships

### Model Outputs
- Trained PCA, CCA, and GMM models (pickled)
- Component loadings and canonical variates
- Cluster statistics (means, covariances, mixing coefficients)

### Reports
- Summary statistics for each cluster
- Correlation matrices between original variables and components
- Validation metrics (silhouette score, Davies-Bouldin index)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). Springer.
- Hardoon, D. R., Szedmák, S., & Shawe-Taylor, J. (2004). Canonical correlation analysis: An overview with application to learning methods. Neural Computation, 16(12), 2639-2664.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
- Von Storch, H., & Zwiers, F. W. (1999). Statistical Analysis in Climate Research. Cambridge University Press.
- Vaittinada Ayar, P., Battisti, D. S., Li, C., King, M., Vrac, M., & Tjiputra, J. (2023). A regime view of ENSO flavors through clustering in CMIP6 models. *Earth's Future*, 11, e2022EF003460. https://doi.org/10.1029/2022EF003460
- Schlör, J., Strnad, F., Capotondi, A., & Goswami, B. (2024). Contribution of El Niño Southern Oscillation (ENSO) diversity to low-frequency changes in ENSO variance. *Geophysical Research Letters*, 51, e2024GL109179. https://doi.org/10.1029/2024GL109179

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.
