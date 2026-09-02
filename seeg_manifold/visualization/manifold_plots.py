"""
Manifold Visualization Functions

Tools for visualizing dimensionality reduction results and manifold structure.
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
import warnings


def plot_embedding_2d(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    title: str = "2D Embedding",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    alpha: float = 0.7,
    s: int = 30,
    colorbar_label: str = "",
    ax=None,
    **kwargs
):
    """
    Plot 2D embedding.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding (n_samples, 2)
    labels : np.ndarray, optional
        Discrete labels for coloring points
    colors : np.ndarray, optional
        Continuous values for coloring points
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    alpha : float
        Point transparency
    s : int
        Point size
    colorbar_label : str
        Label for colorbar
    ax : matplotlib axis, optional
        Existing axis to plot on
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    
    if embedding.shape[1] != 2:
        raise ValueError(f"Expected 2D embedding, got shape {embedding.shape}")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Determine coloring
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                label=str(label), alpha=alpha, s=s, **kwargs
            )
        ax.legend()
    elif colors is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=colors, cmap=cmap, alpha=alpha, s=s, **kwargs
        )
        cbar = plt.colorbar(scatter, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            alpha=alpha, s=s, **kwargs
        )
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    
    return fig, ax


def plot_embedding_3d(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    title: str = "3D Embedding",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    alpha: float = 0.7,
    s: int = 30,
    elev: float = 30,
    azim: float = 45,
    ax=None,
    **kwargs
):
    """
    Plot 3D embedding.
    
    Parameters
    ----------
    embedding : np.ndarray
        3D embedding (n_samples, 3)
    labels : np.ndarray, optional
        Discrete labels for coloring points
    colors : np.ndarray, optional
        Continuous values for coloring points
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    alpha : float
        Point transparency
    s : int
        Point size
    elev : float
        Elevation angle
    azim : float
        Azimuth angle
    ax : matplotlib axis, optional
        Existing 3D axis to plot on
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if embedding.shape[1] != 3:
        raise ValueError(f"Expected 3D embedding, got shape {embedding.shape}")
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Determine coloring
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                label=str(label), alpha=alpha, s=s, **kwargs
            )
        ax.legend()
    elif colors is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            c=colors, cmap=cmap, alpha=alpha, s=s, **kwargs
        )
        plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            alpha=alpha, s=s, **kwargs
        )
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    
    return fig, ax


def plot_embedding_comparison(
    embeddings: Dict[str, np.ndarray],
    colors: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 12),
    cmap: str = 'viridis',
    **kwargs
):
    """
    Plot comparison of multiple embeddings.
    
    Parameters
    ----------
    embeddings : dict
        Dictionary mapping method names to embeddings
    colors : np.ndarray, optional
        Continuous values for coloring points
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
        
    Returns
    -------
    fig
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    n_methods = len(embeddings)
    
    # Determine if 2D or 3D
    first_embed = list(embeddings.values())[0]
    is_3d = first_embed.shape[1] >= 3
    
    # Calculate grid layout
    ncols = min(n_methods, 2)
    nrows = (n_methods + ncols - 1) // ncols
    
    if is_3d:
        fig = plt.figure(figsize=figsize)
        for i, (method, embedding) in enumerate(embeddings.items()):
            ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
            
            if colors is not None:
                ax.scatter(
                    embedding[:, 0], embedding[:, 1], embedding[:, 2],
                    c=colors, cmap=cmap, alpha=0.7, s=20
                )
            else:
                ax.scatter(
                    embedding[:, 0], embedding[:, 1], embedding[:, 2],
                    alpha=0.7, s=20
                )
            
            ax.set_title(method.upper())
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        
        for i, (method, embedding) in enumerate(embeddings.items()):
            ax = axes[i]
            
            if colors is not None:
                ax.scatter(
                    embedding[:, 0], embedding[:, 1],
                    c=colors, cmap=cmap, alpha=0.7, s=20
                )
            else:
                ax.scatter(
                    embedding[:, 0], embedding[:, 1],
                    alpha=0.7, s=20
                )
            
            ax.set_title(method.upper())
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
        
        # Hide unused axes
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_trajectory(
    embedding: np.ndarray,
    time_points: Optional[np.ndarray] = None,
    line_alpha: float = 0.3,
    point_alpha: float = 0.7,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    title: str = "Trajectory",
    **kwargs
):
    """
    Plot embedding as trajectory with time coloring.
    
    Parameters
    ----------
    embedding : np.ndarray
        Embedding (n_samples, 2 or 3)
    time_points : np.ndarray, optional
        Time values for coloring (default: sample indices)
    line_alpha : float
        Transparency of connecting lines
    point_alpha : float
        Transparency of points
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    title : str
        Plot title
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    
    n_dim = embedding.shape[1]
    
    if time_points is None:
        time_points = np.arange(embedding.shape[0])
    
    if n_dim == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trajectory line
        ax.plot(embedding[:, 0], embedding[:, 1], 
                'k-', alpha=line_alpha, linewidth=0.5)
        
        # Plot points colored by time
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=time_points, cmap=cmap, alpha=point_alpha, s=30
        )
        plt.colorbar(scatter, ax=ax, label='Time')
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        
    elif n_dim >= 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory line
        ax.plot(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                'k-', alpha=line_alpha, linewidth=0.5)
        
        # Plot points colored by time
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            c=time_points, cmap=cmap, alpha=point_alpha, s=30
        )
        plt.colorbar(scatter, ax=ax, label='Time', shrink=0.6)
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(title)
    else:
        raise ValueError(f"Expected 2D or 3D embedding, got {n_dim}D")
    
    return fig, ax


def plot_scree(
    explained_variance_ratio: np.ndarray,
    n_components: Optional[int] = None,
    threshold: float = 0.95,
    figsize: Tuple[int, int] = (12, 4),
    title: str = "PCA Scree Plot"
):
    """
    Plot PCA scree plot with cumulative variance.
    
    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Explained variance ratio for each component
    n_components : int, optional
        Number of components to show (default: min(30, total))
    threshold : float
        Variance threshold to mark
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    fig
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    if n_components is None:
        n_components = min(30, len(explained_variance_ratio))
    
    var_ratio = explained_variance_ratio[:n_components]
    cumvar = np.cumsum(var_ratio)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scree plot
    axes[0].bar(range(1, len(var_ratio)+1), var_ratio, alpha=0.7)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Variance')
    
    # Cumulative variance
    axes[1].plot(range(1, len(cumvar)+1), cumvar, 'b-o', markersize=4)
    axes[1].axhline(threshold, color='r', linestyle='--', 
                    label=f'{threshold*100:.0f}% threshold')
    
    # Mark where threshold is reached
    n_for_threshold = np.argmax(cumvar >= threshold) + 1
    axes[1].axvline(n_for_threshold, color='g', linestyle='--',
                    label=f'{n_for_threshold} components')
    
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance')
    axes[1].legend()
    axes[1].set_xlim([0, n_components+1])
    axes[1].set_ylim([0, 1.05])
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig

