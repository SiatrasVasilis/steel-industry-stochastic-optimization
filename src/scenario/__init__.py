"""
Scenario Module

This module contains scenario generation tools for stochastic optimization.

Classes
-------
BaseScenarioGenerator
    Abstract base class for all scenario generators (from base.py)
VarModelScenarioGenerator
    VAR-based scenario generation engine (from var.py)
RegimeSwitchingGenerator
    Markov-Switching VAR scenario generation (from regime_switching.py)

Dataclasses
-----------
GeneratorConfig, StressConfig
    Core configuration dataclasses (from base.py)
ShockDistribution
    Enum for shock distribution types (from base.py)
GenerationResult, ReductionResult, AnalysisResult, OrderSelectionResult,
DiagnosticResult, StationarityResult
    Result containers (from base.py)
"""

from .base import (
    BaseScenarioGenerator,
    GeneratorConfig,
    StressConfig,
    ShockDistribution,
    GenerationResult,
    ReductionResult,
    AnalysisResult,
    OrderSelectionResult,
    DiagnosticResult,
    StationarityResult,
    FitDiagnosticsReport,
)
from .var import VarModelScenarioGenerator
from .regime_switching import RegimeSwitchingGenerator

__all__ = [
    # Base class
    'BaseScenarioGenerator',
    # Config & enums
    'GeneratorConfig',
    'StressConfig',
    'ShockDistribution',
    # Result containers
    'GenerationResult',
    'ReductionResult',
    'AnalysisResult',
    'OrderSelectionResult',
    'DiagnosticResult',
    'StationarityResult',
    'FitDiagnosticsReport',
    # VAR generator
    'VarModelScenarioGenerator',
    # Regime-Switching generator
    'RegimeSwitchingGenerator',
]
