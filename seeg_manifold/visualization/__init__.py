"""
Visualization Module

Tools for visualizing manifold learning results and neural data:
- Static plots with matplotlib
- Interactive 3D plots with plotly
- Trajectory visualization
- Comparison plots
"""

from .manifold_plots import (
    plot_embedding_2d,
    plot_embedding_3d,
    plot_embedding_comparison,
    plot_trajectory,
    plot_scree,
)

from .interactive import (
    plot_embedding_interactive,
    plot_trajectory_interactive,
    plot_comparison_interactive,
)

__all__ = [
    # Static plots
    'plot_embedding_2d',
    'plot_embedding_3d',
    'plot_embedding_comparison',
    'plot_trajectory',
    'plot_scree',
    # Interactive plots
    'plot_embedding_interactive',
    'plot_trajectory_interactive',
    'plot_comparison_interactive',
]

