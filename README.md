# SEEG Manifold & Symmetry Analysis Toolkit

A Python toolkit for finding low-dimensional structure in intracranial EEG
(SEEG/iEEG) recordings, and for testing whether that structure carries
geometric symmetry — with permutation tests rather than eyeballed plots.

## Motivation

Population neural activity is high-dimensional by construction — one dimension
per electrode — but the activity that varies meaningfully often lies on a
much lower-dimensional manifold. Two problems follow. First, the estimate of
"how many dimensions" depends heavily on the method and on how the data matrix
is oriented, so a single number from a single estimator is not trustworthy.
Second, once an embedding is computed it is tempting to read symmetry off the
picture: a plot that *looks* six-fold will happily be described as six-fold.

That second failure mode is the one this toolkit is built around. A detector
that reports the strongest rotational harmonic will always return some fold,
including for isotropic noise. The symmetry functions here therefore compare
the observed structure against a null distribution and return a permutation
p-value, so "no symmetry" is an available answer.

## Method

The pipeline has four stages:

1. **Preprocess** — Butterworth band-pass and notch filtering (zero-phase,
   `sosfiltfilt`), common average reference, threshold- or ICA-based artifact
   handling, and fixed-length or event-locked epoching. Epoch rejection uses a
   median/MAD criterion so that clean recordings are not thrown away and a
   single large artifact cannot mask itself by inflating the statistics.

2. **Estimate intrinsic dimensionality** — three independent estimators
   (PCA variance threshold, PCA elbow via maximum curvature, and the
   Levina–Bickel maximum-likelihood estimator), reported individually plus a
   median consensus. Disagreement between them is diagnostic, not noise.

3. **Embed and compare** — PCA, UMAP, t-SNE, Isomap and PHATE behind one
   interface, each scored by how well it preserves the original pairwise
   geometry (Spearman correlation of distances) and, optionally, local
   neighbourhoods (trustworthiness and continuity).

4. **Test for structure** — representational dissimilarity matrices,
   persistent homology (H0/H1/H2, Betti curves, persistence landscapes and
   images), and symmetry detection: rotational (k-fold), reflection,
   periodicity, and translation invariance.

The angular structure of an embedding is binned and Fourier transformed;
k-fold symmetry shows up as power at harmonic k. That power is compared
against surrogates — uniformly distributed angles for rotation, angle-randomised
points with preserved radii for reflection — and the resulting p-value is
Bonferroni corrected across the folds tested.

## Installation

```bash
git clone https://github.com/Jarmck-Cai/eeg_manifold.git
cd eeg_manifold

conda env create -f environment.yml
conda activate seeg-manifold
pip install -e .
```

Or with pip only:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional analyses (UMAP, PHATE, persistent homology, interactive plots) are not
required for the core pipeline:

```bash
pip install -e ".[full]"
```

Tested on Python 3.11 with NumPy 2.4, SciPy 1.17, scikit-learn 1.9 and MNE 1.12.
**MNE must be ≥ 1.6**: earlier versions call `np.cast`, which NumPy 2.0 removed.

## Usage

The quick-start notebook runs the whole pipeline on synthetic data with a known
ground truth, so every number it prints can be checked against the value it
should recover. No external data is needed.

```bash
jupyter lab notebooks/00_quick_start.ipynb
```

Runtime is about 1–2 minutes on a laptop CPU; no GPU is used anywhere in this
repository. The notebook is checked in with its outputs, so the results below
are visible without running anything.

A minimal programmatic example:

```python
import numpy as np
from seeg_manifold.preprocessing import preprocess_pipeline
from seeg_manifold.preprocessing.epoching import concatenate_epochs
from seeg_manifold.manifold import estimate_dimensionality, fit_pca
from seeg_manifold.symmetry import detect_rotational_symmetry

data = np.random.randn(32, 30000)        # (n_channels, n_timepoints)

processed, info = preprocess_pipeline(data, sfreq=500.0,
                                      lowcut=1, highcut=100, notch_freq=50)

# Estimators expect (n_samples, n_features): timepoints x channels.
samples = concatenate_epochs(processed).T
dims = estimate_dimensionality(samples[:4000])
print(dims['consensus'])

embedding = fit_pca(samples[:4000], n_components=2)
result = detect_rotational_symmetry(embedding, n_permutations=1000)
if result['significant']:
    print(f"{result['best_fold']}-fold, p = {result['best_p_value']:.4f}")
else:
    print("no significant rotational symmetry")
```

To load your own recording instead:

```python
from seeg_manifold.io import load_seeg_data

data = load_seeg_data('my_recording.mat')   # .mat, .edf or .fif
processed, info = preprocess_pipeline(data.data, sfreq=data.sfreq)
```

## Results

All numbers below are produced by `notebooks/00_quick_start.ipynb` and are
stored in its committed outputs. They come from synthetic data, and are
included to show the methods behaving correctly on a case where the right
answer is known — not as findings about real brains.

**Dimensionality recovery.** Five oscillatory latent sources are mixed into 32
channels through a random mixing matrix, plus independent per-channel noise.
The signal subspace is exactly 5-dimensional:

| Estimator | Estimate |
|---|---|
| PCA (95% variance) | 5 |
| PCA (elbow) | 6 |
| Levina–Bickel MLE | 4.3 |
| **Consensus (median)** | **5** |
| *True latent dimensionality* | *5* |

**Embedding quality**, as Spearman correlation between pairwise distances
before and after embedding into 3 dimensions (1500 timepoints):

| Method | Distance correlation |
|---|---|
| PCA | 0.916 |
| t-SNE | 0.562 |
| Isomap | 0.544 |
| UMAP | 0.476 |

PCA scores highest because the underlying structure is a linear subspace.
t-SNE and UMAP optimise local neighbourhoods rather than global distances, so
lower values here are expected behaviour, not failure.

**Symmetry detection with a negative control.** Six conditions whose mean
patterns sit at six evenly spaced angles in a 2D coding plane, against a
matched dataset with the same trial count and noise but no condition structure:

| Dataset | Best fold | p (corrected) | Significant |
|---|---|---|---|
| Structured (6 conditions) | 6 | 0.007 | yes |
| Control (no structure) | 8 | 1.000 | no |

The control is the point of the table: the detector names a fold for it too,
and only the p-value distinguishes that from the real thing.

Reproduce all of the above with:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/00_quick_start.ipynb
```

## Repository structure

```
seeg_manifold/
├── io/              # SEEGData container; .mat (v7 and v7.3), .edf, .fif loaders
├── preprocessing/   # filtering, artifact removal, epoching, unified pipeline
├── features/        # spectral (PSD, band power, entropy) and connectivity
├── manifold/        # dimensionality estimation, reduction, method comparison
├── rsa/             # representational dissimilarity matrices
├── topology/        # persistent homology, Betti curves, landscapes, images
├── symmetry/        # rotational, reflection, periodicity, translation tests
└── visualization/   # matplotlib and plotly embedding plots

config/              # default parameters (preprocessing section is loaded)
notebooks/           # 00_quick_start.ipynb, committed with outputs
tests/               # pytest suite
```

## Data

**No data is included in this repository, and none is required.** The
quick-start notebook and the entire test suite run on synthetic data generated
in-process.

For your own recordings, `load_seeg_data` accepts `.mat` (recommended), `.edf`
and `.fif`. Export from MATLAB as:

```matlab
seeg_data = struct();
seeg_data.data = your_data;       % (n_channels x n_timepoints)
seeg_data.sfreq = 1000;           % sampling rate
seeg_data.ch_names = ch_names;    % cell array of channel names
save('my_seeg_data.mat', 'seeg_data', '-v7.3');
```

SEEG recordings are clinical data. Users are responsible for the ethical
approvals, de-identification and data-sharing conditions attached to their own
recordings; nothing in this repository handles that for you.

## Status and limitations

Working and tested — `pytest tests/` gives 90 passed, 1 skipped (the PHATE
test, which is skipped unless `phate` is installed):

- Data IO, preprocessing pipeline, epoching
- Dimensionality estimation and all reduction methods
- Spectral and connectivity features
- Symmetry detection, including the permutation tests

Known limitations:

- **Synthetic validation only.** Every number in this README comes from
  synthetic data. The toolkit has not been validated against a public iEEG
  benchmark, and no claim is made about real recordings.
- **Topology and RSA modules are untested.** They are implemented and
  importable but have no test coverage; `topology` additionally requires
  `ripser`/`persim`, and the `wasserstein_distance` wrapper in particular has
  not been exercised against the current `persim` API.
- **PLV on continuous data** is computed across time within a single segment
  rather than across trials, which is a biased estimator. Prefer epoched input.
- **Group-theoretic analysis is not implemented.** Only discrete rotational and
  reflection symmetry are detected. There is no representation-theoretic
  decomposition and no equivariance testing.
- `compute_neighborhood_preservation` builds full pairwise distance matrices
  and is O(n²) in memory; subsample before calling it on large embeddings.
- **Reflection symmetry subsamples by default.** Its permutation test costs
  O(n_permutations × n_angles × n log n), so `detect_reflection_symmetry`
  caps the cloud at `max_points=1000` unless told otherwise.
- The symmetry nulls are covariance-matched Gaussian surrogates, so the tests
  ask whether angular structure exists *beyond* the cloud's second-order
  shape. Symmetry that is purely elliptical is by construction not reported.
- Only the `preprocessing` section of `config/default_config.yaml` is read by
  code; the other sections are documentation of internal defaults.

## Related work

The methods implemented here build on:

- Levina & Bickel (2004), *Maximum Likelihood Estimation of Intrinsic
  Dimension* — the MLE dimensionality estimator.
- Kriegeskorte, Mur & Bandettini (2008), *Representational similarity
  analysis* — the RSA framework.
- McInnes, Healy & Melville (2018), *UMAP: Uniform Manifold Approximation and
  Projection* — the UMAP embedding.
- Moon et al. (2019), *Visualizing structure and transitions in
  high-dimensional biological data* — PHATE.
- Bauer (2021), *Ripser: efficient computation of Vietoris–Rips persistence
  barcodes* — the persistent homology backend.
- Chaudhuri et al. (2019), *The intrinsic attractor manifold and population
  dynamics of a canonical cognitive circuit across waking and sleep* — an
  example of the ring-manifold structure the symmetry tests look for.

## License

MIT — see [LICENSE](LICENSE).
