from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seeg-manifold",
    version="0.1.0",
    author="Jarmck Cai",
    author_email="caizx392@gmail.com",
    description=(
        "Manifold, topology and symmetry analysis of SEEG/iEEG recordings"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jarmck-Cai/eeg_manifold",
    packages=find_packages(include=["seeg_manifold", "seeg_manifold.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24,<3.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        # mne < 1.6 calls np.cast, which NumPy 2.0 removed.
        "mne>=1.6",
        "h5py>=3.8",
        "pyyaml>=6.0",
        "matplotlib>=3.7",
    ],
    extras_require={
        "full": [
            "umap-learn>=0.5.4",
            "phate>=1.0",
            "ripser>=0.6",
            "persim>=0.3",
            "plotly>=5.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
        ],
    },
)
