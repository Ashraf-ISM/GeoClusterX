<div align="center">

# 🌍 GeoClusterX

### Earthquake Catalog Declustering via Unsupervised Machine Learning

*A research-grade thesis workspace for separating background seismicity from clustered events using Self-Organizing Maps, DBSCAN, and complementary methods applied to the New Zealand earthquake catalog.*

---

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20Research-blueviolet)
![Domain](https://img.shields.io/badge/Domain-Seismology%20%7C%20Geophysics-teal)

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Scientific Background](#-scientific-background)
3. [Methods](#-methods)
4. [Repository Structure](#-repository-structure)
5. [Dataset](#-dataset)
6. [Feature Engineering](#-feature-engineering)
7. [Pipeline Architecture](#-pipeline-architecture)
8. [Environment Setup](#-environment-setup)
9. [Usage](#-usage)
10. [Results](#-results)
11. [Evaluation Strategy](#-evaluation-strategy)
12. [Roadmap](#-roadmap)
13. [References](#-references)

---

## 🔭 Project Overview

**GeoClusterX** is a thesis research workspace for earthquake catalog declustering using unsupervised and supervised machine learning. The project targets the **New Zealand seismic catalog** (~380,000 events) and focuses on identifying and removing clustered seismic events — aftershocks, foreshocks, and swarms — to produce a clean background seismicity catalog suitable for:

- Probabilistic seismic hazard analysis (PSHA)
- Gutenberg–Richter *b*-value estimation
- Seismicity rate modeling and forecasting
- Tectonic interpretation

Traditional declustering techniques such as Gardner–Knopoff (1974) and Reasenberg (1985) rely on **empirically derived space–time windows** that may not generalize across different tectonic environments. This project investigates whether **data-driven, machine-learning-based approaches** can achieve robust, physically interpretable declustering without hard-coded thresholds.

---

## 🔬 Scientific Background

Earthquake catalogs contain two statistically distinct populations:

| Population | Characteristics |
|---|---|
| **Background (mainshocks)** | Temporally and spatially independent; Poissonian in time |
| **Clustered events** | Aftershocks, foreshocks, and swarms; spatially and temporally correlated; magnitude-dependent occurrence rates |

Mixing both populations biases seismicity statistics. Declustering aims to isolate the background catalog so that:

- The inter-event time distribution approximates a **Poisson process**
- *b*-value estimates are free from aftershock contamination
- Seismic hazard models assume spatial stationarity

The **Nearest-Neighbour Distance (NND)** method (Zaliapin & Ben-Zion, 2013) provides a rigorous, physics-based framework for quantifying event proximity in a combined space–time–magnitude domain. This project builds on the NND framework and augments it with machine learning to handle large, complex catalogs.

---

## 🧠 Methods

### 1. Self-Organizing Maps (SOM) — `som.py`

SOM is an unsupervised artificial neural network that projects high-dimensional seismic feature vectors onto a **topologically ordered 2-D grid**, preserving neighbourhood relationships in the input space.

**Why SOM for declustering?**
- Captures nonlinear, complex relationships between seismic features
- Provides visual interpretability through component planes and U-matrix
- Adaptable to any catalog size through batch training

**Workflow:**
```
Scaled feature vectors → Train MiniSom grid → Inspect U-matrix →
Assign events to winning neurons → Threshold neurons → Label events
```

---

### 2. DBSCAN — `dbscan.py`

Density-Based Spatial Clustering of Applications with Noise groups events by density in the engineered feature space and marks sparse events as noise — treated as background seismicity.

**Why DBSCAN for declustering?**
- Number of clusters is inferred from data, not specified a priori
- Naturally handles irregular cluster shapes (aftershock sequences)
- Noise class directly maps to background events

**Key hyperparameters:**
- `eps` — neighbourhood radius in scaled feature space
- `min_samples` — minimum events to form a core point

---

### 3. SOM + DBSCAN (Hybrid Pipeline) — `som_dbscan.py`

The primary method in this thesis chains SOM dimensionality reduction with DBSCAN clustering:

```
Raw catalog
    ↓ Feature engineering & scaling
    ↓ SOM training (topological compression)
    ↓ DBSCAN on SOM node coordinates
    ↓ Cluster labels → mainshock/aftershock classification
    ↓ Declustered catalog export
```

This hybrid approach combines the **topology-preserving compression** of SOM with the **noise-robust density clustering** of DBSCAN, resulting in more stable cluster boundaries than either algorithm alone.

---

### 4. HDBSCAN — `experiments/unsupervised/NZ_Method2_HDBSCAN.ipynb`

Hierarchical DBSCAN relaxes the fixed `eps` requirement and operates across multiple density scales. Better suited to catalogs with heterogeneous cluster densities (e.g., shallow crustal vs. subduction interface seismicity).

---

### 5. Gaussian Mixture Models (GMM) — `experiments/unsupervised/NZ_Method3_GMM.ipynb`

Models the joint distribution of seismic features as a mixture of Gaussians. Each component corresponds to a distinct seismic regime (background or cluster). Provides **soft probabilistic cluster assignments**, enabling uncertainty quantification.

---

### 6. KDM — Kohonen Map Declustering Method — `notebooks/unsupervised/som/KDM/`

Direct implementation of the method described in:
> Septier et al. — *"Unsupervised probabilistic machine learning applied to seismicity declustering"*

Adapted to the New Zealand catalog. Uses pre-computed nearest-neighbour temporal (T1–T10) and spatial (R1–R10) distance features alongside magnitude and *b*-value to train a Kohonen map.

---

### 7. Nearest-Neighbour Distance (NND) — `nnd.py`

Physics-based declustering using the **Zaliapin–Ben-Zion** metric:

$$\eta_{ij} = t_{ij} \cdot r_{ij}^{d_f} \cdot 10^{-b M_i}$$

where $t_{ij}$ is inter-event time, $r_{ij}$ is spatial distance, $d_f$ is fractal dimension of seismicity, and $b$ is the Gutenberg–Richter *b*-value. Events are separated in the ($\eta$, $M$) plane into background and clustered populations.

---

### 8. Deep Learning — Autoencoder — `experiments/unsupervised/NZ_DeepLearning_Autoencoder.ipynb`

An autoencoder compresses seismic features into a low-dimensional latent space. Reconstruction error is used as an anomaly score: events with high error are candidate outliers (background seismicity), while those with low error cluster in the latent space.

---

### 9. Fractal Dimension Analysis — `notebooks/unsupervised/fractal_dimension/`

Before applying the NND method, the fractal dimension $d_f$ of the seismic point cloud is estimated via box-counting. This provides a calibrated spatial scaling exponent essential for the NND metric.

---

### 10. GMT Visualization — `scripts/gmt/study-area.sh`

Generic Mapping Tools (GMT) shell scripts produce publication-quality spatial maps of the study area, seismicity distributions, and cluster overlays. Outputs live in `results/figures/unsupervised/gmt/`.

---

## 📁 Repository Structure

```text
GeoClusterX/
│
├── data/                                  # Thesis datasets (excluded from Git if large)
│   ├── supervised/
│   │   ├── raw/                           # Original labeled datasets
│   │   └── processed/                     # Feature-engineered supervised inputs
│   └── unsupervised/
│       ├── raw/                           # Raw earthquake catalog files
│       └── processed/
│           ├── som_dbscan/                # Scaled arrays and intermediate outputs
│           └── gmt/                       # GMT-ready text files
│
├── notebooks/                             # Interactive Jupyter notebooks
│   ├── supervised/                        # (Reserved for future supervised work)
│   └── unsupervised/
│       ├── som/                           # SOM exploration & KDM implementation
│       │   └── KDM/
│       │       └── KDM_NewZealand_Implementation.py
│       ├── som_dbscan/                    # Primary SOM + DBSCAN workflow
│       │   ├── 05_SOM_DBSCAN.ipynb        # Main analysis notebook
│       │   └── FINAL_NZ_Earthquake_Declustering_SOM_DBSCAN.ipynb
│       ├── hdbscan/                       # HDBSCAN exploration
│       └── fractal_dimension/             # Box-counting & fractal dimension
│
├── experiments/                           # Comparison studies and exploratory work
│   ├── supervised/                        # Gradient Boosting, RF, XGB, SVM, K-Means
│   └── unsupervised/
│       ├── DBSCAN/                        # Standalone DBSCAN experiments
│       ├── NZ_Method2_HDBSCAN.ipynb
│       ├── NZ_Method3_GMM.ipynb
│       ├── NZ_Method4_Autoencoder.ipynb
│       ├── NZ_Method5_Comparison.ipynb    # Cross-method benchmarking
│       └── NZ_DeepLearning_Autoencoder.ipynb
│
├── results/                               # All generated thesis outputs
│   ├── figures/
│   │   └── unsupervised/
│   │       ├── som_dbscan/
│   │       └── gmt/
│   ├── tables/
│   ├── reports/
│   │   └── unsupervised/som_dbscan/
│   └── catalogs/
│       └── unsupervised/som_dbscan/       # Declustered catalog (CSV)
│
├── scripts/
│   └── gmt/
│       └── study-area.sh                  # GMT map generation script
│
├── docs/
│   ├── methodology/
│   │   ├── README.md
│   │   └── declustering-earthquake-catalog.md
│   ├── papers/
│   │   └── SOM+DBSCAN.pdf
│   └── repository-roadmap.md
│
├── declustering-earthquake-catalog/       # Reusable Python package (production layer)
│   ├── src/
│   │   └── declustering/
│   │       ├── __init__.py
│   │       ├── preprocess.py              # Catalog loading & feature engineering
│   │       ├── som.py                     # SOM training & labeling
│   │       ├── dbscan.py                  # DBSCAN clustering wrapper
│   │       ├── som_dbscan.py              # End-to-end hybrid pipeline
│   │       ├── nnd.py                     # Nearest-neighbour distance method
│   │       └── utils.py                   # Shared utilities
│   └── scripts/
│       ├── run_som.py                     # CLI runner: SOM only
│       ├── run_dbscan.py                  # CLI runner: DBSCAN only
│       ├── run_som_dbscan.py              # CLI runner: SOM + DBSCAN pipeline
│       └── run_nnd.py                     # CLI runner: NND declustering
│
├── requirements.txt                       # Python dependencies
├── .devcontainer/                         # VS Code Dev Container configuration
└── .gitignore
```

---

## 📂 Dataset

The primary dataset is the **New Zealand GeoNet earthquake catalog**, covering the full national network.

| Field | Description |
|---|---|
| `latitude` | Event latitude (decimal degrees) |
| `longitude` | Event longitude (decimal degrees) |
| `depth` | Hypocentral depth (km) |
| `magnitude` | Event magnitude (Mw / ML / Mb) |
| `time` | Origin time (UTC / decimal years) |
| `bval` | Local Gutenberg–Richter *b*-value |
| `T1–T10` | Temporal distances to 10 nearest neighbours |
| `R1–R10` | Spatial distances to 10 nearest neighbours |
| `Mn` | Mean magnitude feature (KDM) |

**Data placement convention:**

| Stage | Location |
|---|---|
| Raw catalog files | `data/unsupervised/raw/` |
| Cleaned / feature-engineered inputs | `data/unsupervised/processed/<method>/` |
| Final declustered catalogs | `results/catalogs/unsupervised/<method>/` |

> **Note:** Large catalog files (> 50 MB) are excluded from version control via `.gitignore`. Refer to the GeoNet catalog portal for raw data acquisition.

---

## ⚙️ Feature Engineering

The following feature set is derived from the raw catalog and used as input to all ML methods:

| Feature | Category | Description |
|---|---|---|
| Latitude, Longitude | Spatial | Geographic coordinates |
| Depth | Spatial | Hypocentral depth (km) |
| Δt | Temporal | Inter-event time to previous event |
| T1–T10 | Temporal | Distances to 10 nearest temporal neighbours |
| R1–R10 | Spatial | Distances to 10 nearest spatial neighbours |
| Magnitude | Seismic | Event magnitude |
| ΔM | Seismic | Magnitude difference from preceding event |
| *b*-value | Seismic | Local Gutenberg–Richter *b*-value |
| Fractal dim. $d_f$ | Seismic | Box-counting fractal dimension of seismicity |

All features are **Min-Max normalized** to [0, 1] prior to SOM training and DBSCAN application to prevent any single feature from dominating the distance metric.

---

## 🔄 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Raw Earthquake Catalog                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Preprocessing  (preprocess.py)              │
│   • Remove null/inconsistent entries                     │
│   • Compute inter-event times and ΔM                     │
│   • Compute k-NN temporal & spatial distances            │
│   • Estimate fractal dimension (box-counting)            │
│   • Estimate local b-value (sliding window MLE)          │
│   • Min-Max feature scaling                              │
└────────────────────────┬────────────────────────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
         SOM          DBSCAN         NND
       (som.py)     (dbscan.py)    (nnd.py)
            │            │            │
            └────────────┼────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│             Cluster Label Assignment                     │
│   • Clustered events  →  aftershocks / foreshocks        │
│   • Noise / background →  mainshocks                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Evaluation & Validation                 │
│   • b-value stability check                              │
│   • Inter-event time distribution (Poisson test)         │
│   • Spatial cluster map (GMT / PyGMT)                    │
│   • Cross-method comparison                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               results/  (thesis outputs)                 │
│   figures/   tables/   reports/   catalogs/              │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Environment Setup

### Option A — pip + virtualenv

```bash
# Clone the repository
git clone https://github.com/Ashraf-ISM/GeoClusterX.git
cd GeoClusterX

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B — conda (recommended for PyGMT)

```bash
conda env create -f declustering-earthquake-catalog/environment.yml
conda activate declustering
```

### Option C — Dev Container (VS Code)

Open the repository in VS Code and select **"Reopen in Container"** when prompted. The `.devcontainer/` configuration installs all dependencies automatically.

### Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Numerical computing & tabular data |
| `scikit-learn` | DBSCAN, preprocessing, metrics |
| `minisom` | Self-Organizing Maps |
| `hdbscan` | Hierarchical DBSCAN |
| `scipy` | Statistical tests, spatial functions |
| `matplotlib`, `seaborn` | Plotting & visualization |
| `pygmt` | Publication-quality GMT maps in Python |
| `jupyter`, `notebook` | Interactive analysis |

---

## 🚀 Usage

### Run the SOM + DBSCAN pipeline

```bash
python declustering-earthquake-catalog/scripts/run_som_dbscan.py \
    --input  data/unsupervised/processed/som_dbscan/catalog_features.csv \
    --output results/catalogs/unsupervised/som_dbscan/
```

### Run DBSCAN standalone

```bash
python declustering-earthquake-catalog/scripts/run_dbscan.py \
    --input  data/unsupervised/processed/catalog_scaled.csv \
    --eps 0.3 --min-samples 5
```

### Run NND declustering

```bash
python declustering-earthquake-catalog/scripts/run_nnd.py \
    --input  data/unsupervised/raw/nz_catalog.csv \
    --b-value 1.0 --fractal-dim 1.6
```

### Run SOM training

```bash
python declustering-earthquake-catalog/scripts/run_som.py \
    --input  data/unsupervised/processed/catalog_scaled.csv \
    --grid-x 10 --grid-y 10 --iterations 5000
```

### Open the primary analysis notebook

```bash
jupyter notebook notebooks/unsupervised/som_dbscan/05_SOM_DBSCAN.ipynb
```

### Generate GMT study-area map

```bash
bash scripts/gmt/study-area.sh
```

---

## 📊 Results

Generated thesis outputs are organized by method under `results/`:

| Output type | Location |
|---|---|
| Spatial cluster maps (SOM + DBSCAN) | `results/figures/unsupervised/som_dbscan/` |
| GMT seismicity maps | `results/figures/unsupervised/gmt/` |
| Summary statistics tables | `results/tables/unsupervised/` |
| Declustering reports | `results/reports/unsupervised/som_dbscan/` |
| Declustered catalog (CSV) | `results/catalogs/unsupervised/som_dbscan/` |

Key visual outputs from the SOM + DBSCAN workflow include:

- **Fig 1** — Spatial overview of the New Zealand catalog with cluster overlays (`Fig1_spatial_overview.png`)
- **Fig 2** — Classification diagnostics: U-matrix, component planes, DBSCAN labels (`Fig2_classification_diagnostics.png`)
- NZ catalog labelled analysis (`NZ_catalog_labelled_analysis.png`)

---

## 🧪 Evaluation Strategy

Because declustering is an **unsupervised problem**, ground-truth labels are not available. Evaluation relies on a combination of physical and statistical tests:

| Test | What it measures |
|---|---|
| *b*-value stability | Gutenberg–Richter *b*-value should stabilize after declustering |
| Inter-event time distribution | Residual catalog should approximate a Poisson process (exponential inter-event times) |
| Spatial independence | Absence of spatial clustering in the background catalog |
| Comparison with classical methods | Cross-validation against Gardner–Knopoff and Reasenberg results |
| Cross-method benchmarking | `NZ_Method5_Comparison.ipynb` compares SOM+DBSCAN, HDBSCAN, GMM, Autoencoder, and KDM |

---

## 🗺️ Roadmap

See [`docs/repository-roadmap.md`](docs/repository-roadmap.md) for the full plan. Key milestones:

- [x] Repository structure established (supervised / unsupervised separation)
- [x] Primary SOM + DBSCAN notebook (`05_SOM_DBSCAN.ipynb`)
- [x] KDM implementation for New Zealand catalog
- [x] HDBSCAN, GMM, and Autoencoder experiments
- [x] GMT study-area map generation
- [x] Fractal dimension estimation notebooks
- [x] Reusable Python package skeleton (`declustering-earthquake-catalog/`)
- [ ] Promote stable notebook logic into versioned package modules
- [ ] CLI scripts with full argument parsing
- [ ] Dataset manifest (provenance, coverage, preprocessing assumptions)
- [ ] Reproduce all thesis figures from scripts only (no notebooks)
- [ ] Supervised declustering experiments (Random Forest, XGBoost, SVM)
- [ ] Automated evaluation report generation

---

## 📖 References

1. **Gardner, J. K. & Knopoff, L.** (1974). Is the sequence of earthquakes in Southern California, with aftershocks removed, Poissonian? *Bulletin of the Seismological Society of America*, 64(5), 1363–1367.
2. **Reasenberg, P.** (1985). Second-order moment of central California seismicity. *Journal of Geophysical Research*, 90(B7), 5479–5495.
3. **Zaliapin, I. & Ben-Zion, Y.** (2013). Earthquake clusters in southern California I: Identification and stability. *Journal of Geophysical Research: Solid Earth*, 118(6), 2847–2864.
4. **Septier, F. et al.** Unsupervised probabilistic machine learning applied to seismicity declustering. *(see `docs/papers/SOM+DBSCAN.pdf`)*
5. **Vesanto, J. & Alhoniemi, E.** (2000). Clustering of the Self-Organizing Map. *IEEE Transactions on Neural Networks*, 11(3), 586–600.
6. **Ester, M. et al.** (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96 Proceedings*, 226–231.
7. **Campello, R. J. G. B. et al.** (2013). Density-based clustering based on hierarchical density estimates. *PAKDD*, 7819, 160–172.

---

## 🤝 Contributing

This is an active thesis repository. Contributions, suggestions, and issues are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please follow the **Working Convention** below when adding new material:

| Material | Location |
|---|---|
| Raw catalog files | `data/<mode>/raw/` |
| Cleaned / engineered data | `data/<mode>/processed/` |
| Interactive analysis | `notebooks/<mode>/` |
| Comparison studies | `experiments/<mode>/` |
| Generated figures, tables, reports | `results/.../<mode>/` |
| Stable, reusable logic | `declustering-earthquake-catalog/src/declustering/` |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built for reproducible geophysical research · New Zealand Seismology · Machine Learning*

</div>
