"""
Parameters Module

This module contains parameter dataclasses for optimization models.

Classes
-------
SimpleModelParams
    Parameters for SimpleStochasticModel (business/cost parameters)
RiskProfile
    Risk preferences for stochastic optimization (risk_aversion, cvar_alpha)
"""

from .risk import RiskProfile

__all__ = [
    'RiskProfile',
]
