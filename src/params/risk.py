"""Risk Profile Parameters

This module contains the RiskProfile dataclass for managing risk-related
optimization parameters.

Classes
-------
RiskProfile
    Dataclass containing risk preferences for stochastic optimization.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import json


@dataclass
class RiskProfile:
    """
    Risk preferences for stochastic optimization.
    
    This dataclass encapsulates risk-related parameters that control the
    trade-off between expected value maximization and downside risk protection.
    
    Attributes
    ----------
    risk_aversion : float
        Risk aversion parameter λ for CVaR objective weighting.
        Range: [0, 1]. 
        - 0 = risk-neutral (maximize expected value only)
        - 1 = fully risk-averse (maximize CVaR only)
        - 0.3 = moderate risk aversion (70% expected value, 30% CVaR)
        Default: 0.0 (risk-neutral)
    cvar_alpha : float
        CVaR tail probability (quantile level).
        Range: (0, 1). Smaller values focus on more extreme tails.
        - 0.05 = 5% worst-case scenarios (95% CVaR)
        - 0.10 = 10% worst-case scenarios (90% CVaR)
        Default: 0.05
        
    Examples
    --------
    >>> # Risk-neutral profile (default)
    >>> risk = RiskProfile()
    >>> 
    >>> # Moderate risk aversion
    >>> risk = RiskProfile(risk_aversion=0.3)
    >>> 
    >>> # Conservative profile focusing on worst 10%
    >>> risk = RiskProfile(risk_aversion=0.5, cvar_alpha=0.10)
    >>> 
    >>> # Use with optimizer
    >>> model = StochasticOptimizationModel(solver="highs")
    >>> model.optimize(scenarios, prob, params, risk)
    
    Notes
    -----
    The optimization objective becomes:
        (1 - λ) * E[profit] + λ * CVaR_α[profit]
    
    Where:
    - λ = risk_aversion
    - α = cvar_alpha
    - CVaR_α is the Conditional Value-at-Risk at level α
    """
    
    risk_aversion: float = 0.3
    cvar_alpha: float = 0.05
    
    def __post_init__(self):
        """Validate risk parameters after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate parameter values.
        
        Raises
        ------
        ValueError
            If any parameter violates its constraints.
        """
        errors = []
        
        if self.risk_aversion < 0 or self.risk_aversion > 1:
            errors.append(f"risk_aversion must be in [0, 1], got {self.risk_aversion}")
        if self.cvar_alpha <= 0 or self.cvar_alpha >= 1:
            errors.append(f"cvar_alpha must be in (0, 1), got {self.cvar_alpha}")
        
        if errors:
            raise ValueError("Invalid risk parameters:\n  - " + "\n  - ".join(errors))
    
    @property
    def is_risk_neutral(self) -> bool:
        """Check if this profile is risk-neutral (risk_aversion == 0)."""
        return self.risk_aversion == 0.0
    
    @property
    def is_fully_risk_averse(self) -> bool:
        """Check if this profile is fully risk-averse (risk_aversion == 1)."""
        return self.risk_aversion == 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert risk profile to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskProfile":
        """Create risk profile from dictionary."""
        return cls(**data)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export risk profile to JSON.
        
        Parameters
        ----------
        filepath : str, optional
            If provided, write to file. Otherwise return JSON string.
            
        Returns
        -------
        str
            JSON representation of risk profile.
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_json(cls, filepath_or_str: str) -> "RiskProfile":
        """
        Load risk profile from JSON file or string.
        
        Parameters
        ----------
        filepath_or_str : str
            File path or JSON string.
            
        Returns
        -------
        RiskProfile
            Loaded risk profile.
        """
        try:
            with open(filepath_or_str, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, OSError):
            data = json.loads(filepath_or_str)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        return f"RiskProfile(risk_aversion={self.risk_aversion}, cvar_alpha={self.cvar_alpha})"
    
    # =========================================================================
    # Predefined profiles (convenience factory methods)
    # =========================================================================
    
    @classmethod
    def risk_neutral(cls) -> "RiskProfile":
        """Create a risk-neutral profile (maximizes expected value only)."""
        return cls(risk_aversion=0.0)
    
    @classmethod
    def conservative(cls) -> "RiskProfile":
        """Create a conservative risk profile (moderate risk aversion)."""
        return cls(risk_aversion=0.3, cvar_alpha=0.05)
    
    @classmethod
    def aggressive(cls) -> "RiskProfile":
        """Create an aggressive risk profile (low risk aversion)."""
        return cls(risk_aversion=0.1, cvar_alpha=0.10)
    
    @classmethod
    def defensive(cls) -> "RiskProfile":
        """Create a defensive risk profile (high risk aversion)."""
        return cls(risk_aversion=0.5, cvar_alpha=0.05)
