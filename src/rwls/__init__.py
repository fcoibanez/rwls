"""Regime-weighted least squares (RWLS) for regime-dependent factor models.

Latent market regimes are identified with a multivariate Gaussian hidden Markov
model fitted to common factor returns. Each asset's factor loadings and residual
variance are then estimated separately for every regime by weighted least squares,
with each observation weighted by its posterior regime probability.
"""

from rwls.estimator import RWLS, AssetResults, RWLSResults, SparseRegimeWarning
from rwls.expanding import Step, expanding
from rwls.factors import load_factors
from rwls.moments import RegimeMoments
from rwls.regimes import RegimeFit, RegimeModel, RegimeSelection, select_regimes
from rwls.settings import RegimeSettings, RWLSSettings

__version__ = "0.1.0"

__all__ = [
    "AssetResults",
    "RWLS",
    "RWLSResults",
    "RegimeFit",
    "RegimeModel",
    "RegimeMoments",
    "RegimeSelection",
    "RegimeSettings",
    "RWLSSettings",
    "SparseRegimeWarning",
    "Step",
    "expanding",
    "load_factors",
    "select_regimes",
    "__version__",
]
