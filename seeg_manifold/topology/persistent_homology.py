"""
Persistent Homology for Topological Data Analysis

Compute and analyze persistent homology of point clouds.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Union
import warnings


def compute_persistence_diagram(
    data: np.ndarray,
    max_dim: int = 1,
    max_edge_length: Optional[float] = None,
    metric: str = 'euclidean'
) -> List[np.ndarray]:
    """
    Compute persistence diagram using Ripser.
    
    Parameters
    ----------
    data : np.ndarray
        Point cloud (n_points, n_features)
    max_dim : int
        Maximum homology dimension to compute
    max_edge_length : float, optional
        Maximum edge length for Rips complex
    metric : str
        Distance metric
        
    Returns
    -------
    diagrams : list of np.ndarray
        Persistence diagrams for each dimension
        Each diagram is array of (birth, death) pairs
    """
    try:
        import ripser
    except ImportError:
        raise ImportError("Ripser required. Install with: pip install ripser")
    
    # Flatten if 3D
    if data.ndim == 3:
        n_samples = data.shape[0]
        data = data.reshape(n_samples, -1)
    
    # Compute persistence
    kwargs = {
        'maxdim': max_dim,
        'metric': metric
    }
    if max_edge_length is not None:
        kwargs['thresh'] = max_edge_length
    
    result = ripser.ripser(data, **kwargs)
    
    return result['dgms']


def compute_persistence_landscape(
    diagram: np.ndarray,
    num_landscapes: int = 5,
    resolution: int = 100,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> np.ndarray:
    """
    Compute persistence landscapes from a persistence diagram.
    
    Parameters
    ----------
    diagram : np.ndarray
        Persistence diagram (n_points, 2) with (birth, death) pairs
    num_landscapes : int
        Number of landscape functions to compute
    resolution : int
        Number of sample points
    min_val : float, optional
        Minimum x value
    max_val : float, optional
        Maximum x value
        
    Returns
    -------
    landscapes : np.ndarray
        Landscape functions (num_landscapes, resolution)
    """
    # Remove infinite death times
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]
    
    if len(diagram) == 0:
        return np.zeros((num_landscapes, resolution))
    
    if min_val is None:
        min_val = np.min(diagram[:, 0])
    if max_val is None:
        max_val = np.max(diagram[:, 1])
    
    x = np.linspace(min_val, max_val, resolution)
    
    # Compute tent functions for each point
    tent_functions = np.zeros((len(diagram), resolution))
    
    for i, (birth, death) in enumerate(diagram):
        mid = (birth + death) / 2
        height = (death - birth) / 2
        
        # Rising part: birth to mid
        mask_rise = (x >= birth) & (x <= mid)
        tent_functions[i, mask_rise] = x[mask_rise] - birth
        
        # Falling part: mid to death
        mask_fall = (x > mid) & (x <= death)
        tent_functions[i, mask_fall] = death - x[mask_fall]
    
    # Compute landscapes as sorted values
    landscapes = np.zeros((num_landscapes, resolution))
    
    for j in range(resolution):
        values = np.sort(tent_functions[:, j])[::-1]
        for k in range(min(num_landscapes, len(values))):
            landscapes[k, j] = values[k]
    
    return landscapes


def compute_betti_curve(
    diagram: np.ndarray,
    resolution: int = 100,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Betti curve from persistence diagram.
    
    Parameters
    ----------
    diagram : np.ndarray
        Persistence diagram
    resolution : int
        Number of sample points
    min_val : float, optional
        Minimum filtration value
    max_val : float, optional
        Maximum filtration value
        
    Returns
    -------
    filtration_values : np.ndarray
        Filtration parameter values
    betti_numbers : np.ndarray
        Betti numbers at each filtration value
    """
    # Remove infinite death times for range calculation
    finite_mask = np.isfinite(diagram[:, 1])
    finite_diagram = diagram[finite_mask]
    
    if len(finite_diagram) == 0:
        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = 1
        return np.linspace(min_val, max_val, resolution), np.zeros(resolution)
    
    if min_val is None:
        min_val = np.min(diagram[:, 0])
    if max_val is None:
        max_val = np.max(finite_diagram[:, 1])
    
    filtration_values = np.linspace(min_val, max_val, resolution)
    betti_numbers = np.zeros(resolution)
    
    for i, t in enumerate(filtration_values):
        # Count features alive at time t
        alive = (diagram[:, 0] <= t) & (diagram[:, 1] > t)
        betti_numbers[i] = np.sum(alive)
    
    return filtration_values, betti_numbers


def persistence_statistics(
    diagrams: List[np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics for persistence diagrams.
    
    Parameters
    ----------
    diagrams : list of np.ndarray
        Persistence diagrams for each dimension
        
    Returns
    -------
    stats : dict
        Statistics for each dimension
    """
    stats = {}
    
    for dim, diagram in enumerate(diagrams):
        # Remove infinite points
        finite_mask = np.isfinite(diagram[:, 1])
        finite_diagram = diagram[finite_mask]
        
        if len(finite_diagram) == 0:
            stats[f'H{dim}'] = {
                'n_features': 0,
                'mean_persistence': 0,
                'max_persistence': 0,
                'total_persistence': 0
            }
            continue
        
        lifetimes = finite_diagram[:, 1] - finite_diagram[:, 0]
        
        stats[f'H{dim}'] = {
            'n_features': len(finite_diagram),
            'mean_persistence': float(np.mean(lifetimes)),
            'max_persistence': float(np.max(lifetimes)),
            'total_persistence': float(np.sum(lifetimes)),
            'std_persistence': float(np.std(lifetimes)),
            'mean_birth': float(np.mean(finite_diagram[:, 0])),
            'mean_death': float(np.mean(finite_diagram[:, 1]))
        }
    
    return stats


def bottleneck_distance(
    diagram1: np.ndarray,
    diagram2: np.ndarray
) -> float:
    """
    Compute bottleneck distance between two persistence diagrams.
    
    Parameters
    ----------
    diagram1 : np.ndarray
        First persistence diagram
    diagram2 : np.ndarray
        Second persistence diagram
        
    Returns
    -------
    distance : float
        Bottleneck distance
    """
    try:
        import persim
    except ImportError:
        raise ImportError("Persim required. Install with: pip install persim")
    
    return persim.bottleneck(diagram1, diagram2)


def wasserstein_distance(
    diagram1: np.ndarray,
    diagram2: np.ndarray,
    matching: bool = False
):
    """
    Compute Wasserstein distance between two persistence diagrams.

    Parameters
    ----------
    diagram1 : np.ndarray
        First persistence diagram
    diagram2 : np.ndarray
        Second persistence diagram
    matching : bool
        If True, also return the optimal matching between the diagrams.

    Returns
    -------
    distance : float
        Wasserstein distance, or ``(distance, matching)`` when
        ``matching=True``.

    Notes
    -----
    ``persim.wasserstein`` computes the 1-Wasserstein (optimal transport)
    distance and exposes no order parameter, so none is offered here.
    """
    try:
        import persim
    except ImportError:
        raise ImportError("Persim required. Install with: pip install persim")

    return persim.wasserstein(diagram1, diagram2, matching=matching)


def persistence_image(
    diagram: np.ndarray,
    resolution: Tuple[int, int] = (50, 50),
    sigma: float = 0.1,
    birth_range: Optional[Tuple[float, float]] = None,
    persistence_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Compute persistence image from diagram.
    
    Parameters
    ----------
    diagram : np.ndarray
        Persistence diagram
    resolution : tuple
        Image resolution (height, width)
    sigma : float
        Gaussian kernel bandwidth
    birth_range : tuple, optional
        Range for birth axis
    persistence_range : tuple, optional
        Range for persistence axis
        
    Returns
    -------
    image : np.ndarray
        Persistence image
    """
    # Remove infinite points
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]
    
    if len(diagram) == 0:
        return np.zeros(resolution)
    
    # Convert to birth-persistence coordinates
    births = diagram[:, 0]
    persistences = diagram[:, 1] - diagram[:, 0]
    
    if birth_range is None:
        birth_range = (births.min(), births.max())
    if persistence_range is None:
        persistence_range = (0, persistences.max())
    
    # Create grid
    x = np.linspace(birth_range[0], birth_range[1], resolution[1])
    y = np.linspace(persistence_range[0], persistence_range[1], resolution[0])
    X, Y = np.meshgrid(x, y)
    
    # Compute image as sum of Gaussians weighted by persistence
    image = np.zeros(resolution)
    
    for birth, persistence in zip(births, persistences):
        # Gaussian kernel
        gaussian = np.exp(-((X - birth)**2 + (Y - persistence)**2) / (2 * sigma**2))
        # Weight by persistence (more persistent = more important)
        image += persistence * gaussian
    
    return image

