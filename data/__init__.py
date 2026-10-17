"""
Data handling, automated patch selection, and synthetic error generation modules.
"""

from .dataset import SARDataset, generate_synthetic_2d_phase_error

__all__ = [
    'SARDataset',
    'generate_synthetic_2d_phase_error',
]
