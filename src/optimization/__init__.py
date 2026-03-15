"""
Optimization Module

This module contains optimization models for steel industry planning.

Classes
-------
BaseStochasticModel
    Abstract base class for all optimization models (from base.py).
    Provides variable mapping, input validation, and the ``run()``
    convenience method.
StochasticOptimizationModel
    Two-stage stochastic LP with CVaR for procurement and capacity planning (from stochastic.py)
SafetyStockModel
    Benchmark model with traditional safety stock planning (from benchmark.py)
SafetyStockParams
    Policy parameters for the safety stock model.
OptimizationResult
    Container for optimization output (decisions, risk metrics, scenarios).
BacktestResult
    Container for backtest output (timeline, metrics, plots).

Functions
---------
compare_decisions
    Side-by-side Plotly comparison of two OptimizationResult objects.
"""

from .base import BaseStochasticModel
from .stochastic import StochasticOptimizationModel
from .benchmark import SafetyStockModel, SafetyStockParams
from .results import OptimizationResult, BacktestResult, compare_decisions

__all__ = [
    'BaseStochasticModel',
    'StochasticOptimizationModel',
    'SafetyStockModel',
    'SafetyStockParams',
    'OptimizationResult',
    'BacktestResult',
    'compare_decisions',
]
