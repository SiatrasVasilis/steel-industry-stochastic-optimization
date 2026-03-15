"""Data Loader for Steel Industry Time Series

This module provides a reusable DataLoader class for loading, preprocessing,
and preparing time series data for scenario generation and optimization models.

Supports multiple data sources:
- FRED API (Federal Reserve Economic Data)
- CSV files
- Direct DataFrame input

Classes
-------
DataLoader
    Main data loading and preprocessing class.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Optional, Union, List, Tuple
from pathlib import Path


class DataLoader:
    """
    Data loader and preprocessor for steel industry time series.
    
    Handles loading raw data from various sources, preprocessing (resampling,
    alignment, missing values), and transformations (log returns, subsetting)
    needed before fitting scenario generators.
    
    Parameters
    ----------
    fred_api_key : str, optional
        FRED API key for loading data from the Federal Reserve Economic Data API.
        Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
    variable_names : Dict[str, str], optional
        Mapping of internal variable names to descriptive labels.
        Default: {'P': 'Steel Price', 'C': 'Scrap Cost', 'D': 'Demand'}
    
    Attributes
    ----------
    raw_data : pd.DataFrame or None
        Raw loaded data before preprocessing
    data : pd.DataFrame or None
        Processed data ready for model fitting
    log_returns_data : pd.DataFrame or None
        Log returns of the processed data
    
    Examples
    --------
    >>> # Load from FRED API
    >>> loader = DataLoader(fred_api_key="your_key")
    >>> loader.load_from_fred()
    >>> loader.subset(n_observations=180, last_observation='2019-12-01')
    >>> log_ret = loader.compute_log_returns()
    
    >>> # Load from CSV
    >>> loader = DataLoader()
    >>> loader.load_from_csv("data/steel_prices.csv", date_column="Date")
    
    >>> # Load from existing DataFrame
    >>> loader = DataLoader()
    >>> loader.load_from_dataframe(my_df)
    
    >>> # Method chaining
    >>> data = (DataLoader(fred_api_key="key")
    ...     .load_from_fred()
    ...     .subset(n_observations=180)
    ...     .compute_log_returns()
    ...     .get_log_returns())
    """
    
    # Default FRED series for steel industry
    DEFAULT_FRED_SERIES: Dict[str, str] = {
        'P': 'WPU101704',   # PPI: Hot Rolled Steel Bars, Plates, and Structural Shapes
        'C': 'WPU1012',     # PPI: Iron and Steel Scrap
        'D': 'IPG3311A2S',  # Industrial Production: Steel Products
    }
    
    # Alternative FRED series that users might want
    ALTERNATIVE_FRED_SERIES: Dict[str, Dict[str, str]] = {
        'P': {
            'WPU101702': 'PPI: Steel Mill Products',
            'WPU10170211': 'PPI: Hot Rolled Steel Sheet and Strip',
            'WPU101704': 'PPI: Hot Rolled Bars, Plates, Structural Shapes',
        },
        'C': {
            'WPU1012': 'PPI: Iron and Steel Scrap',
            'WPU101201': 'PPI: Ferrous Scrap',
            'WPSSOP1200': 'PPI: Scrap and Waste Materials',
        },
        'D': {
            'IPG3311A2S': 'Industrial Production: Steel Products',
            'IPG331111CN': 'Industrial Production: Iron and Steel Mills',
            'ISRG3311A2S': 'Capacity Utilization: Steel Products',
        },
    }
    
    DEFAULT_VARIABLE_LABELS: Dict[str, str] = {
        'P': 'Steel Price',
        'C': 'Scrap Cost',
        'D': 'Demand',
    }
    
    _log_level: str = "INFO"
    
    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        variable_labels: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        fred_api_key : str, optional
            FRED API key. Can also be loaded from environment variable FRED_API_KEY.
        variable_labels : Dict[str, str], optional
            Custom variable name → label mapping.
        """
        self.fred_api_key = fred_api_key or self._try_load_api_key()
        self.variable_labels = variable_labels or self.DEFAULT_VARIABLE_LABELS.copy()
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.data: Optional[pd.DataFrame] = None
        self.log_returns_data: Optional[pd.DataFrame] = None
        
        # Metadata
        self._source: Optional[str] = None
        self._fred_series_used: Optional[Dict[str, str]] = None
        self._subset_params: Optional[Dict] = None
        self._real_prices_config: Optional[Dict] = None  # Real price conversion config
    
    # =========================================================================
    # Data Loading Methods
    # =========================================================================
    
    def load_from_fred(
        self,
        series_ids: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        plot_data: bool = False,
    ) -> "DataLoader":
        """
        Load time series data from the FRED API.
        
        Parameters
        ----------
        series_ids : Dict[str, str], optional
            Mapping of variable name → FRED series ID.
            Default uses DEFAULT_FRED_SERIES:
            {'P': 'WPU101704', 'C': 'WPU1012', 'D': 'IPG3311A2S'}
        start_date : str, optional
            Start date filter (format: 'YYYY-MM-DD'). None = all available.
        end_date : str, optional
            End date filter (format: 'YYYY-MM-DD'). None = most recent.
        plot_data : bool, default True
            Whether to plot the loaded time series.
            
        Returns
        -------
        self : DataLoader
            Returns self for method chaining.
            
        Raises
        ------
        ValueError
            If no FRED API key is available.
        ConnectionError
            If FRED API is unreachable.
        """
        from fredapi import Fred
        
        if self.fred_api_key is None:
            raise ValueError(
                "FRED API key is required. Pass it to DataLoader(fred_api_key='...') "
                "or set the FRED_API_KEY environment variable."
            )
        
        logger = self._setup_logger()
        series_ids = series_ids or self.DEFAULT_FRED_SERIES.copy()
        self._fred_series_used = series_ids
        
        logger.info(f"Loading data from FRED API...")
        fred = Fred(api_key=self.fred_api_key)
        
        series_data = {}
        for var_name, series_id in series_ids.items():
            logger.debug(f"  Fetching {var_name}: {series_id}")
            raw = fred.get_series(series_id).to_frame(name=var_name)
            raw.index = pd.to_datetime(raw.index)
            
            # Apply date filters
            if start_date:
                raw = raw.loc[pd.to_datetime(start_date):]
            if end_date:
                raw = raw.loc[:pd.to_datetime(end_date)]
            
            # Resample to monthly frequency
            series_data[var_name] = raw.resample("MS").mean()
        
        # Merge and align all series
        data = pd.concat(series_data.values(), axis=1).dropna()
        
        self.raw_data = data.copy()
        self.data = data.copy()
        self._source = "fred"
        
        # Reset derived data
        self.log_returns_data = None
        self._subset_params = None
        
        logger.info(
            f"[✓] Loaded {len(data)} observations "
            f"({data.index.min().strftime('%Y-%m')} to {data.index.max().strftime('%Y-%m')})"
        )
        
        if plot_data:
            self._plot_time_series(data, title="Raw Data from FRED API")
        
        return self
    
    def load_from_csv(
        self,
        filepath: Union[str, Path],
        date_column: str = "Date",
        column_mapping: Optional[Dict[str, str]] = None,
        freq: str = "MS",
        plot_data: bool = True,
    ) -> "DataLoader":
        """
        Load time series data from a CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.
        date_column : str, default "Date"
            Name of the date column.
        column_mapping : Dict[str, str], optional
            Mapping from CSV column names to internal variable names.
            E.g., {'steel_price': 'P', 'scrap_price': 'C', 'demand': 'D'}
        freq : str, default "MS"
            Target frequency for resampling. "MS" = month start.
        plot_data : bool, default True
            Whether to plot the loaded data.
            
        Returns
        -------
        self : DataLoader
            Returns self for method chaining.
        """
        logger = self._setup_logger()
        logger.info(f"Loading data from CSV: {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=[date_column])
        df = df.set_index(date_column).sort_index()
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required = list(self.variable_labels.keys())
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Available: {list(df.columns)}. "
                f"Use column_mapping to rename columns."
            )
        
        # Keep only required columns and resample
        df = df[required]
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df = df.resample(freq).mean().dropna()
        
        self.raw_data = df.copy()
        self.data = df.copy()
        self._source = "csv"
        self.log_returns_data = None
        self._subset_params = None
        
        logger.info(
            f"[✓] Loaded {len(df)} observations "
            f"({df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')})"
        )
        
        if plot_data:
            self._plot_time_series(df, title=f"Data from {Path(filepath).name}")
        
        return self
    
    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        column_mapping: Optional[Dict[str, str]] = None,
        plot_data: bool = False,
    ) -> "DataLoader":
        """
        Load data from an existing pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex and variable columns.
        column_mapping : Dict[str, str], optional
            Mapping from DataFrame column names to internal names.
        plot_data : bool, default False
            Whether to plot the loaded data.
            
        Returns
        -------
        self : DataLoader
            Returns self for method chaining.
        """
        logger = self._setup_logger()
        
        df = df.copy()
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")
        
        required = list(self.variable_labels.keys())
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        
        df = df[required].dropna()
        
        self.raw_data = df.copy()
        self.data = df.copy()
        self._source = "dataframe"
        self.log_returns_data = None
        self._subset_params = None
        
        logger.info(f"[✓] Loaded {len(df)} observations from DataFrame")
        
        if plot_data:
            self._plot_time_series(df, title="Data from DataFrame")
        
        return self
    
    # =========================================================================
    # Preprocessing Methods
    # =========================================================================
    
    def subset(
        self,
        n_observations: Optional[int] = None,
        last_observation: Optional[Union[str, int]] = None,
        lag_order: int = 2,
        plot_data: bool = False,
    ) -> "DataLoader":
        """
        Extract a subset of historical data for model estimation.
        
        Parameters
        ----------
        n_observations : int, optional
            Number of observations to keep. If None, uses empirical rule: 90 * lag_order.
        last_observation : str or int, optional
            Last observation to include. Useful for backtesting.
            - str/datetime: treated as date cutoff
            - int: treated as index position
            - None: use all available data
        lag_order : int, default 2
            Intended lag order (used for automatic n_observations calculation).
        plot_data : bool, default False
            Whether to plot the subset.
            
        Returns
        -------
        self : DataLoader
            Returns self for method chaining.
        """
        self._check_data_loaded()
        logger = self._setup_logger()
        
        if n_observations is None:
            n_observations = 90 * lag_order
            logger.debug(f"Auto n_observations: 90 × {lag_order} = {n_observations}")
        
        # Start from raw data to allow re-subsetting
        working_data = self.raw_data.copy()
        
        # Apply last_observation cutoff
        if last_observation is not None:
            if isinstance(last_observation, int):
                working_data = working_data.iloc[:last_observation + 1]
            else:
                working_data = working_data.loc[:pd.to_datetime(last_observation)]
        
        # Take last n observations
        if n_observations > len(working_data):
            logger.warning(
                f"Requested {n_observations} observations but only {len(working_data)} available. "
                f"Using all {len(working_data)}."
            )
        
        working_data = working_data.tail(n_observations)
        self.data = working_data
        self.log_returns_data = None  # Reset derived data
        
        self._subset_params = {
            'n_observations': n_observations,
            'last_observation': last_observation,
            'lag_order': lag_order,
            'actual_observations': len(working_data),
        }
        
        logger.info(
            f"[✓] Subset: {len(working_data)} observations "
            f"({working_data.index.min().strftime('%Y-%m')} to "
            f"{working_data.index.max().strftime('%Y-%m')})"
        )
        
        if plot_data:
            self._plot_time_series(
                working_data, 
                title=f"Estimation Data ({len(working_data)} observations)"
            )
        
        return self
    
    def convert_to_real_prices(
        self,
        anchor_date: str,
        real_data: Dict[str, float],
        plot_data: bool = False,
    ) -> "DataLoader":
        """
        Convert index data to real price units using an anchor date.
        
        This method scales the index data (e.g., FRED PPI indices) to real-world
        units (e.g., $/ton) based on known prices at a specific anchor date.
        The conversion preserves the relative movements in the data.
        
        Parameters
        ----------
        anchor_date : str
            Date when the known prices were observed (format: 'YYYY-MM-DD').
            This date must exist in the data.
        real_data : Dict[str, float]
            Real prices at the anchor date.
            E.g., {'P': 520.0, 'C': 130.0, 'D': 50000.0} for $/ton (P, C) and units (D).
        plot_data : bool, default False
            Whether to plot the converted data.
            
        Returns
        -------
        self : DataLoader
            Returns self for method chaining.
            
        Examples
        --------
        >>> loader = (
        ...     DataLoader()
        ...     .load_from_fred()
        ...     .subset(n_observations=180, last_observation='2019-12-01')
        ...     .convert_to_real_prices(
        ...         anchor_date='2019-12-01',
        ...         real_data={'P': 520, 'C': 130, 'D': 50000}
        ...     )
        ...     .compute_log_returns()
        ... )
        
        Notes
        -----
        The conversion formula for each variable is:
        
            real_value = (index_value / anchor_index) * anchor_real_price
        
        This ensures that at the anchor date, the value equals the known price,
        and all other values are scaled proportionally.
        
        Log returns are invariant to this scaling, so this conversion is purely
        for interpretability and visualization.
        """
        self._check_data_loaded()
        logger = self._setup_logger()
        
        anchor_dt = pd.to_datetime(anchor_date)
        
        # Validate anchor date exists in data
        if anchor_dt not in self.data.index:
            # Find closest date
            closest = self.data.index[np.abs(self.data.index - anchor_dt).argmin()]
            logger.warning(
                f"Exact anchor date {anchor_date} not found. "
                f"Using closest date: {closest.strftime('%Y-%m-%d')}"
            )
            anchor_dt = closest
        
        # Get anchor index values
        anchor_values = self.data.loc[anchor_dt]
        
        # Compute conversion factors
        conversion_factors = {}
        for var, real_price in real_data.items():
            if var not in self.data.columns:
                raise ValueError(f"Variable '{var}' not found in data. Available: {list(self.data.columns)}")
            
            anchor_index = anchor_values[var]
            conversion_factors[var] = real_price / anchor_index
        
        # Apply conversion
        converted_data = self.data.copy()
        for var, factor in conversion_factors.items():
            converted_data[var] = converted_data[var] * factor
        
        # Also convert raw_data if it exists
        if self.raw_data is not None:
            converted_raw = self.raw_data.copy()
            for var, factor in conversion_factors.items():
                if var in converted_raw.columns:
                    converted_raw[var] = converted_raw[var] * factor
            self.raw_data = converted_raw
        
        self.data = converted_data
        self.log_returns_data = None  # Reset - needs recomputation
        
        # Store conversion config
        self._real_prices_config = {
            'anchor_date': anchor_date,
            'prices': real_data,
            'conversion_factors': conversion_factors,
            'anchor_index_values': anchor_values.to_dict(),
        }
        
        logger.info(
            f"[✓] Converted to real prices (anchor: {anchor_date})"
        )
        for var, real_price in real_data.items():
            logger.debug(f"    {var}: {anchor_values[var]:.2f} (index) → {real_price:.2f} (real)")
        
        if plot_data:
            self._plot_time_series(
                converted_data,
                title=f"Data in Real Units (anchor: {anchor_date})"
            )
        
        return self

    def compute_log_returns(self, plot_data: bool = False) -> "DataLoader":
        """
        Compute log returns (Δlog) of the time series data.
        
        Log returns approximate percentage changes and are commonly used
        for VAR and other time series models.
        
        Parameters
        ----------
        plot_data : bool, default False
            Whether to plot the log returns and their distributions.
            
        Returns
        -------
        self : DataLoader
            Returns self for method chaining.
        """
        self._check_data_loaded()
        logger = self._setup_logger()
        
        log_data = np.log(self.data)
        self.log_returns_data = log_data.diff().dropna()
        
        logger.info(f"[✓] Log returns computed: {len(self.log_returns_data)} observations")
        
        if plot_data:
            self._plot_log_returns()
        
        return self
    
    # =========================================================================
    # Data Access Methods
    # =========================================================================
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the current processed data.
        
        Returns
        -------
        pd.DataFrame
            Current processed DataFrame (after subsetting, if applied).
            
        Raises
        ------
        RuntimeError
            If no data has been loaded.
        """
        self._check_data_loaded()
        return self.data.copy()
    
    def get_raw_data(self) -> pd.DataFrame:
        """
        Get the original raw loaded data (before subsetting).
        
        Returns
        -------
        pd.DataFrame
            Raw data as originally loaded.
        """
        if self.raw_data is None:
            raise RuntimeError("No data loaded. Call a load_from_* method first.")
        return self.raw_data.copy()
    
    def get_log_returns(self) -> pd.DataFrame:
        """
        Get log returns data.
        
        Returns
        -------
        pd.DataFrame
            Log returns DataFrame.
            
        Raises
        ------
        RuntimeError
            If log returns have not been computed.
        """
        if self.log_returns_data is None:
            raise RuntimeError("Log returns not computed. Call compute_log_returns() first.")
        return self.log_returns_data.copy()
    
    def get_future_data(self, last_observation: str) -> pd.DataFrame:
        """
        Get actual data after the last_observation date (for backtesting).
        
        Parameters
        ----------
        last_observation : str
            Date cutoff. Returns data AFTER this date.
            
        Returns
        -------
        pd.DataFrame
            Future/actual data after the cutoff date.
        """
        if self.raw_data is None:
            raise RuntimeError("No data loaded. Call a load_from_* method first.")
        cutoff = pd.to_datetime(last_observation)
        future = self.raw_data.loc[self.raw_data.index > cutoff]
        if future.empty:
            raise ValueError(f"No data available after {last_observation}")
        return future.copy()
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot(self, what: str = "data", **kwargs) -> None:
        """
        Plot data.
        
        Parameters
        ----------
        what : str
            What to plot: "data", "raw", "log_returns", "all"
        **kwargs
            Additional keyword arguments passed to matplotlib.
        """
        if what in ("data", "all"):
            self._check_data_loaded()
            self._plot_time_series(self.data, title="Processed Data", **kwargs)
        
        if what in ("raw", "all"):
            if self.raw_data is not None:
                self._plot_time_series(self.raw_data, title="Raw Data", **kwargs)
        
        if what in ("log_returns", "all"):
            if self.log_returns_data is not None:
                self._plot_log_returns(**kwargs)
    
    # =========================================================================
    # Summary & Info
    # =========================================================================
    
    def summary(self) -> Dict:
        """
        Get a summary of the DataLoader state.
        
        Returns
        -------
        dict
            Summary information including source, shape, date range, etc.
        """
        info = {
            'source': self._source,
            'variables': list(self.variable_labels.keys()),
            'raw_data_shape': self.raw_data.shape if self.raw_data is not None else None,
            'data_shape': self.data.shape if self.data is not None else None,
            'has_log_returns': self.log_returns_data is not None,
        }
        
        if self.data is not None:
            info['date_range'] = (
                self.data.index.min().strftime('%Y-%m-%d'),
                self.data.index.max().strftime('%Y-%m-%d'),
            )
            info['n_observations'] = len(self.data)
        
        if self._fred_series_used:
            info['fred_series'] = self._fred_series_used
        
        if self._subset_params:
            info['subset_params'] = self._subset_params
        
        if self._real_prices_config:
            info['real_prices_config'] = self._real_prices_config
        
        return info
    
    # =========================================================================
    # Private Helpers
    # =========================================================================
    
    def _check_data_loaded(self) -> None:
        """Raise if no data has been loaded."""
        if self.data is None:
            raise RuntimeError(
                "No data loaded. Call load_from_fred(), load_from_csv(), "
                "or load_from_dataframe() first."
            )
    
    @staticmethod
    def _try_load_api_key() -> Optional[str]:
        """Try to load FRED API key from environment."""
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        return os.environ.get("FRED_API_KEY")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self._log_level, logging.INFO))
        logger.handlers = []
        
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, self._log_level, logging.INFO))
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def _plot_time_series(self, data: pd.DataFrame, title: str = "Time Series", **kwargs) -> None:
        """Plot time series data with shared x-axis."""
        n_vars = len(data.columns)
        fig, axes = plt.subplots(n_vars, 1, figsize=kwargs.get('figsize', (14, 3 * n_vars)), sharex=True)
        
        if n_vars == 1:
            axes = [axes]
        
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
        
        for i, col in enumerate(data.columns):
            ax = axes[i]
            color = colors[i % len(colors)]
            label = self.variable_labels.get(col, col)
            ax.plot(data.index, data[col], color=color, linewidth=1.2)
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{label} ({col})", fontsize=10, fontweight='bold')
        
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()
    
    def _plot_log_returns(self, **kwargs) -> None:
        """Plot log returns: time series and distributions with multiple theoretical fits."""
        if self.log_returns_data is None:
            return
        
        from scipy import stats
        
        data = self.log_returns_data
        n_vars = len(data.columns)
        fig, axes = plt.subplots(n_vars, 2, figsize=kwargs.get('figsize', (14, 3.5 * n_vars)))
        
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['#2196F3', '#FF5722', '#4CAF50']
        
        # Distribution styles for theoretical fits
        dist_styles = {
            'Normal': {'color': '#1a1a1a', 'linestyle': '-', 'linewidth': 2},
            'Student-t': {'color': '#e74c3c', 'linestyle': '--', 'linewidth': 2},
            'Skew-Normal': {'color': '#9b59b6', 'linestyle': '-.', 'linewidth': 2},
        }
        
        for i, col in enumerate(data.columns):
            color = colors[i % len(colors)]
            label = self.variable_labels.get(col, col)
            series = data[col].dropna().values
            
            # Time series
            axes[i, 0].plot(data.index, data[col], color=color, linewidth=0.8, alpha=0.8)
            axes[i, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[i, 0].set_ylabel(f'Δlog({col})')
            axes[i, 0].set_title(f'{label} - Log Returns', fontsize=10)
            axes[i, 0].grid(True, alpha=0.3)
            
            # Distribution histogram
            axes[i, 1].hist(series, bins=40, color=color, alpha=0.5, 
                           edgecolor='white', density=True, label='Empirical')
            axes[i, 1].set_title(f'{label} - Distribution Comparison', fontsize=10)
            axes[i, 1].set_xlabel(f'Δlog({col})')
            
            # X values for PDF plotting
            x = np.linspace(series.min() - 0.01, series.max() + 0.01, 200)
            
            # Fit and plot Normal
            mu, sigma = series.mean(), series.std()
            axes[i, 1].plot(x, stats.norm.pdf(x, mu, sigma),
                          label=f'Normal', **dist_styles['Normal'])
            
            # Fit and plot Student-t
            try:
                df_t, loc_t, scale_t = stats.t.fit(series)
                axes[i, 1].plot(x, stats.t.pdf(x, df_t, loc_t, scale_t),
                              label=f'Student-t (df={df_t:.1f})', **dist_styles['Student-t'])
            except Exception:
                pass
            
            # Fit and plot Skew-Normal
            try:
                a_sn, loc_sn, scale_sn = stats.skewnorm.fit(series)
                axes[i, 1].plot(x, stats.skewnorm.pdf(x, a_sn, loc_sn, scale_sn),
                              label=f'Skew-Normal (α={a_sn:.2f})', **dist_styles['Skew-Normal'])
            except Exception:
                pass
            
            axes[i, 1].legend(fontsize=8, loc='upper right')
            axes[i, 1].grid(True, alpha=0.3)
        
        fig.suptitle("Log Returns Analysis - Distribution Comparison", fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()
    
    def distribution_fit_summary(self) -> pd.DataFrame:
        """
        Compute goodness-of-fit statistics for different distributions.
        
        Returns
        -------
        pd.DataFrame
            Fit statistics (AIC, BIC, log-likelihood) for each variable and distribution.
        """
        if self.log_returns_data is None:
            raise RuntimeError("Log returns not computed. Call compute_log_returns() first.")
        
        from scipy import stats
        
        results = []
        
        for col in self.log_returns_data.columns:
            series = self.log_returns_data[col].dropna().values
            n = len(series)
            
            # Normal fit
            mu, sigma = series.mean(), series.std()
            ll_norm = np.sum(stats.norm.logpdf(series, mu, sigma))
            k_norm = 2  # parameters: mu, sigma
            aic_norm = 2 * k_norm - 2 * ll_norm
            bic_norm = k_norm * np.log(n) - 2 * ll_norm
            
            results.append({
                'variable': col,
                'distribution': 'Normal',
                'params': f'μ={mu:.4f}, σ={sigma:.4f}',
                'log_likelihood': ll_norm,
                'AIC': aic_norm,
                'BIC': bic_norm,
            })
            
            # Student-t fit
            try:
                df_t, loc_t, scale_t = stats.t.fit(series)
                ll_t = np.sum(stats.t.logpdf(series, df_t, loc_t, scale_t))
                k_t = 3  # parameters: df, loc, scale
                aic_t = 2 * k_t - 2 * ll_t
                bic_t = k_t * np.log(n) - 2 * ll_t
                
                results.append({
                    'variable': col,
                    'distribution': 'Student-t',
                    'params': f'df={df_t:.2f}, loc={loc_t:.4f}, scale={scale_t:.4f}',
                    'log_likelihood': ll_t,
                    'AIC': aic_t,
                    'BIC': bic_t,
                })
            except Exception:
                pass
            
            # Skew-Normal fit
            try:
                a_sn, loc_sn, scale_sn = stats.skewnorm.fit(series)
                ll_sn = np.sum(stats.skewnorm.logpdf(series, a_sn, loc_sn, scale_sn))
                k_sn = 3  # parameters: alpha, loc, scale
                aic_sn = 2 * k_sn - 2 * ll_sn
                bic_sn = k_sn * np.log(n) - 2 * ll_sn
                
                results.append({
                    'variable': col,
                    'distribution': 'Skew-Normal',
                    'params': f'α={a_sn:.2f}, loc={loc_sn:.4f}, scale={scale_sn:.4f}',
                    'log_likelihood': ll_sn,
                    'AIC': aic_sn,
                    'BIC': bic_sn,
                })
            except Exception:
                pass
        
        return pd.DataFrame(results)
    
    def __repr__(self) -> str:
        if self.data is not None:
            n = len(self.data)
            src = self._source or "unknown"
            return f"DataLoader(source='{src}', n_observations={n}, variables={list(self.data.columns)})"
        return "DataLoader(no data loaded)"
