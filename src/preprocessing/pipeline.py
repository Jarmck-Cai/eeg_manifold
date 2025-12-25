"""
Preprocessing Pipeline

Unified pipeline for SEEG data preprocessing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import yaml
from pathlib import Path

from .filters import filter_data
from .artifact_removal import remove_artifacts_threshold
from .epoching import epoch_data


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Filtering
    lowcut: float = 1.0
    highcut: float = 150.0
    notch_freq: Optional[float] = 50.0
    notch_width: float = 2.0
    filter_order: int = 4
    
    # Artifact removal
    artifact_method: str = 'threshold'  # 'threshold', 'ica', or 'none'
    artifact_threshold_std: float = 5.0
    
    # Epoching
    epoch_length: float = 2.0
    epoch_overlap: float = 0.5
    reject_threshold: Optional[float] = 5.0
    
    # Common average reference
    apply_car: bool = True
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'PreprocessingConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract preprocessing section if present
        if 'preprocessing' in config_dict:
            config_dict = config_dict['preprocessing']
        
        # Flatten nested structure
        flat_config = {}
        if 'filter' in config_dict:
            flat_config.update(config_dict['filter'])
        if 'artifact' in config_dict:
            for k, v in config_dict['artifact'].items():
                flat_config[f'artifact_{k}'] = v
        if 'epoching' in config_dict:
            for k, v in config_dict['epoching'].items():
                if k == 'epoch_length':
                    flat_config['epoch_length'] = v
                elif k == 'overlap':
                    flat_config['epoch_overlap'] = v
        
        return cls(**{k: v for k, v in flat_config.items() 
                     if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'lowcut': self.lowcut,
            'highcut': self.highcut,
            'notch_freq': self.notch_freq,
            'notch_width': self.notch_width,
            'filter_order': self.filter_order,
            'artifact_method': self.artifact_method,
            'artifact_threshold_std': self.artifact_threshold_std,
            'epoch_length': self.epoch_length,
            'epoch_overlap': self.epoch_overlap,
            'reject_threshold': self.reject_threshold,
            'apply_car': self.apply_car
        }


def preprocess_pipeline(
    data: np.ndarray,
    sfreq: float,
    config: Optional[PreprocessingConfig] = None,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    notch_freq: Optional[float] = None,
    epoch_length: Optional[float] = None,
    epoch_overlap: Optional[float] = None,
    apply_car: Optional[bool] = None,
    return_epochs: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply complete preprocessing pipeline to SEEG data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (n_channels, n_timepoints)
    sfreq : float
        Sampling frequency in Hz
    config : PreprocessingConfig, optional
        Configuration object. Individual parameters override config.
    lowcut : float, optional
        High-pass cutoff frequency
    highcut : float, optional
        Low-pass cutoff frequency
    notch_freq : float, optional
        Notch filter frequency (set to None to disable)
    epoch_length : float, optional
        Epoch length in seconds
    epoch_overlap : float, optional
        Epoch overlap ratio (0-1)
    apply_car : bool, optional
        Whether to apply common average reference
    return_epochs : bool
        If True, return epoched data; if False, return continuous
    verbose : bool
        Print progress information
        
    Returns
    -------
    processed : np.ndarray
        Preprocessed data
    info : dict
        Processing information and parameters used
        
    Examples
    --------
    >>> # Basic preprocessing with defaults
    >>> processed, info = preprocess_pipeline(data, sfreq=1000)
    
    >>> # Custom parameters
    >>> processed, info = preprocess_pipeline(data, sfreq=1000,
    ...                                        lowcut=0.5, highcut=200,
    ...                                        notch_freq=60,
    ...                                        epoch_length=1.0)
    
    >>> # Using config file
    >>> config = PreprocessingConfig.from_yaml('config/default_config.yaml')
    >>> processed, info = preprocess_pipeline(data, sfreq=1000, config=config)
    """
    # Initialize config
    if config is None:
        config = PreprocessingConfig()
    
    # Override with explicit parameters
    if lowcut is not None:
        config.lowcut = lowcut
    if highcut is not None:
        config.highcut = highcut
    if notch_freq is not None:
        config.notch_freq = notch_freq
    if epoch_length is not None:
        config.epoch_length = epoch_length
    if epoch_overlap is not None:
        config.epoch_overlap = epoch_overlap
    if apply_car is not None:
        config.apply_car = apply_car
    
    # Initialize info dict
    info = {
        'config': config.to_dict(),
        'input_shape': data.shape,
        'sfreq': sfreq,
        'steps': []
    }
    
    processed = data.copy()
    
    # Step 1: Filtering
    if verbose:
        print(f"[1/4] Filtering: {config.lowcut}-{config.highcut} Hz, "
              f"notch={config.notch_freq} Hz")
    
    processed = filter_data(
        processed,
        sfreq=sfreq,
        lowcut=config.lowcut,
        highcut=config.highcut,
        notch_freq=config.notch_freq,
        notch_width=config.notch_width,
        order=config.filter_order
    )
    info['steps'].append('filtering')
    
    # Step 2: Common Average Reference
    if config.apply_car:
        if verbose:
            print("[2/4] Applying common average reference")
        processed = processed - np.mean(processed, axis=0, keepdims=True)
        info['steps'].append('car')
    else:
        if verbose:
            print("[2/4] Skipping common average reference")
    
    # Step 3: Artifact removal
    if config.artifact_method != 'none':
        if verbose:
            print(f"[3/4] Artifact removal: {config.artifact_method}")
        
        if config.artifact_method == 'threshold':
            processed, artifact_info = remove_artifacts_threshold(
                processed, sfreq,
                threshold_std=config.artifact_threshold_std,
                return_info=True
            )
            info['artifact_removal'] = artifact_info
        info['steps'].append('artifact_removal')
    else:
        if verbose:
            print("[3/4] Skipping artifact removal")
    
    # Step 4: Epoching
    if return_epochs:
        if verbose:
            print(f"[4/4] Epoching: {config.epoch_length}s epochs, "
                  f"{config.epoch_overlap*100:.0f}% overlap")
        
        processed, times, epoch_info = epoch_data(
            processed, sfreq,
            epoch_length=config.epoch_length,
            overlap=config.epoch_overlap,
            reject_threshold=config.reject_threshold
        )
        info['epoching'] = epoch_info
        info['times'] = times
        info['steps'].append('epoching')
    else:
        if verbose:
            print("[4/4] Skipping epoching (returning continuous data)")
    
    info['output_shape'] = processed.shape
    
    if verbose:
        print(f"\nDone! Output shape: {processed.shape}")
    
    return processed, info
