from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seeg-manifold-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for manifold and symmetry analysis of SEEG data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/seeg_manifold_analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "mne>=1.0.0",
        "scikit-learn>=1.0.0",
        "umap-learn>=0.5.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "full": [
            "phate>=1.0.0",
            "ripser>=0.6.0",
            "persim>=0.3.0",
            "giotto-tda>=0.6.0",
            "pysindy>=1.7.0",
            "geomstats>=2.5.0",
            "plotly>=5.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black",
            "flake8",
        ],
    },
)
