"""
Interactive Visualization Functions

Interactive 3D plots using Plotly for exploring manifold structure.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union


def plot_embedding_interactive(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    hover_data: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Interactive Embedding",
    width: int = 900,
    height: int = 700,
    marker_size: int = 5,
    colorscale: str = 'Viridis',
    opacity: float = 0.8,
    show_legend: bool = True
):
    """
    Create interactive 2D or 3D embedding plot with Plotly.
    
    Parameters
    ----------
    embedding : np.ndarray
        Embedding (n_samples, 2 or 3)
    labels : np.ndarray, optional
        Discrete labels for coloring points
    colors : np.ndarray, optional
        Continuous values for coloring points
    hover_data : dict, optional
        Additional data to show on hover (key: name, value: array)
    title : str
        Plot title
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels
    marker_size : int
        Size of markers
    colorscale : str
        Plotly colorscale name
    opacity : float
        Marker opacity
    show_legend : bool
        Whether to show legend for labeled data
        
    Returns
    -------
    fig
        Plotly figure object
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError("Plotly required. Install with: pip install plotly")
    
    n_dim = embedding.shape[1]
    n_samples = embedding.shape[0]
    
    # Build hover text
    hover_text = None
    if hover_data is not None:
        hover_text = []
        for i in range(n_samples):
            text_parts = [f"Index: {i}"]
            for key, values in hover_data.items():
                text_parts.append(f"{key}: {values[i]}")
            hover_text.append("<br>".join(text_parts))
    
    if n_dim == 2:
        return _plot_2d_interactive(
            embedding, labels, colors, hover_text, title,
            width, height, marker_size, colorscale, opacity, show_legend
        )
    elif n_dim >= 3:
        return _plot_3d_interactive(
            embedding[:, :3], labels, colors, hover_text, title,
            width, height, marker_size, colorscale, opacity, show_legend
        )
    else:
        raise ValueError(f"Embedding must have at least 2 dimensions, got {n_dim}")


def _plot_2d_interactive(
    embedding, labels, colors, hover_text, title,
    width, height, marker_size, colorscale, opacity, show_legend
):
    """Create interactive 2D plot."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            fig.add_trace(go.Scatter(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode='markers',
                name=str(label),
                marker=dict(size=marker_size, opacity=opacity),
                text=np.array(hover_text)[mask] if hover_text else None,
                hoverinfo='text' if hover_text else 'x+y'
            ))
    else:
        marker_dict = dict(
            size=marker_size,
            opacity=opacity
        )
        if colors is not None:
            marker_dict['color'] = colors
            marker_dict['colorscale'] = colorscale
            marker_dict['colorbar'] = dict(title='Value')
        
        fig.add_trace(go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=marker_dict,
            text=hover_text,
            hoverinfo='text' if hover_text else 'x+y'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        width=width,
        height=height,
        showlegend=show_legend and labels is not None
    )
    
    return fig


def _plot_3d_interactive(
    embedding, labels, colors, hover_text, title,
    width, height, marker_size, colorscale, opacity, show_legend
):
    """Create interactive 3D plot."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            fig.add_trace(go.Scatter3d(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                z=embedding[mask, 2],
                mode='markers',
                name=str(label),
                marker=dict(size=marker_size, opacity=opacity),
                text=np.array(hover_text)[mask] if hover_text else None,
                hoverinfo='text' if hover_text else 'x+y+z'
            ))
    else:
        marker_dict = dict(
            size=marker_size,
            opacity=opacity
        )
        if colors is not None:
            marker_dict['color'] = colors
            marker_dict['colorscale'] = colorscale
            marker_dict['colorbar'] = dict(title='Value')
        
        fig.add_trace(go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode='markers',
            marker=marker_dict,
            text=hover_text,
            hoverinfo='text' if hover_text else 'x+y+z'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        width=width,
        height=height,
        showlegend=show_legend and labels is not None
    )
    
    return fig


def plot_trajectory_interactive(
    embedding: np.ndarray,
    time_points: Optional[np.ndarray] = None,
    title: str = "Interactive Trajectory",
    width: int = 900,
    height: int = 700,
    marker_size: int = 4,
    line_width: float = 1.0,
    colorscale: str = 'Viridis'
):
    """
    Create interactive trajectory plot with time coloring.
    
    Parameters
    ----------
    embedding : np.ndarray
        Embedding (n_samples, 2 or 3)
    time_points : np.ndarray, optional
        Time values for coloring (default: sample indices)
    title : str
        Plot title
    width : int
        Figure width
    height : int
        Figure height
    marker_size : int
        Size of markers
    line_width : float
        Width of trajectory line
    colorscale : str
        Plotly colorscale
        
    Returns
    -------
    fig
        Plotly figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly required. Install with: pip install plotly")
    
    n_dim = embedding.shape[1]
    
    if time_points is None:
        time_points = np.arange(embedding.shape[0])
    
    if n_dim == 2:
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='lines',
            line=dict(color='rgba(128, 128, 128, 0.3)', width=line_width),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add points colored by time
        fig.add_trace(go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=time_points,
                colorscale=colorscale,
                colorbar=dict(title='Time'),
                opacity=0.8
            ),
            text=[f"t={t:.2f}" for t in time_points],
            hoverinfo='text+x+y'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            width=width,
            height=height
        )
        
    elif n_dim >= 3:
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode='lines',
            line=dict(color='rgba(128, 128, 128, 0.3)', width=line_width),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add points colored by time
        fig.add_trace(go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=time_points,
                colorscale=colorscale,
                colorbar=dict(title='Time'),
                opacity=0.8
            ),
            text=[f"t={t:.2f}" for t in time_points],
            hoverinfo='text+x+y+z'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            width=width,
            height=height
        )
    else:
        raise ValueError(f"Expected 2D or 3D embedding, got {n_dim}D")
    
    return fig


def plot_comparison_interactive(
    embeddings: Dict[str, np.ndarray],
    colors: Optional[np.ndarray] = None,
    title: str = "Method Comparison",
    height_per_row: int = 400,
    colorscale: str = 'Viridis'
):
    """
    Create interactive comparison of multiple embeddings.
    
    Parameters
    ----------
    embeddings : dict
        Dictionary mapping method names to embeddings
    colors : np.ndarray, optional
        Values for coloring points
    title : str
        Overall title
    height_per_row : int
        Height per row of plots
    colorscale : str
        Plotly colorscale
        
    Returns
    -------
    fig
        Plotly figure object
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly required. Install with: pip install plotly")
    
    methods = list(embeddings.keys())
    n_methods = len(methods)
    
    # Determine layout
    ncols = min(n_methods, 2)
    nrows = (n_methods + ncols - 1) // ncols
    
    # Check dimensionality
    first_embed = embeddings[methods[0]]
    is_3d = first_embed.shape[1] >= 3
    
    if is_3d:
        # Create subplot specs for 3D
        specs = [[{'type': 'scene'} for _ in range(ncols)] for _ in range(nrows)]
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[m.upper() for m in methods],
            specs=specs,
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        for i, method in enumerate(methods):
            row = i // ncols + 1
            col = i % ncols + 1
            
            embedding = embeddings[method][:, :3]
            
            marker_dict = dict(size=3, opacity=0.7)
            if colors is not None:
                marker_dict['color'] = colors
                marker_dict['colorscale'] = colorscale
            
            fig.add_trace(
                go.Scatter3d(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    z=embedding[:, 2],
                    mode='markers',
                    marker=marker_dict,
                    name=method
                ),
                row=row, col=col
            )
    else:
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[m.upper() for m in methods],
            horizontal_spacing=0.08,
            vertical_spacing=0.15
        )
        
        for i, method in enumerate(methods):
            row = i // ncols + 1
            col = i % ncols + 1
            
            embedding = embeddings[method][:, :2]
            
            marker_dict = dict(size=5, opacity=0.7)
            if colors is not None:
                marker_dict['color'] = colors
                marker_dict['colorscale'] = colorscale
            
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    mode='markers',
                    marker=marker_dict,
                    name=method
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title=title,
        height=height_per_row * nrows,
        showlegend=False
    )
    
    return fig

