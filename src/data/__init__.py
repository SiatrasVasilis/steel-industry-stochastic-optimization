"""
Data Module

Provides data loading and preprocessing utilities for scenario generation
and optimization models.

Classes
-------
DataLoader
    Main class for loading data from FRED API, CSV, or DataFrame.
"""

from .loader import DataLoader

__all__ = [
    'DataLoader',
]
