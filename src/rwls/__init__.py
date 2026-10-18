"""Regime-weighted least squares (RWLS) for regime-dependent factor models.

Latent market regimes are identified with a multivariate Gaussian hidden Markov
model fitted to common factor returns. Each asset's factor loadings and residual
variance are then estimated separately for every regime by weighted least squares,
with each observation weighted by its posterior regime probability.

This release reserves the package name. The implementation is under development.
"""

__version__ = "0.0.1"
