"""
Test suite for emotion-driven-markets project.

Tests cover data loading, feature engineering, model training,
and backtesting components to ensure reproducibility.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Project base directory
BASE = Path(__file__).resolve().parents[1]
