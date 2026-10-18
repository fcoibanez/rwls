# rwls

Regime-weighted least squares (RWLS) for estimating regime-dependent factor models.

## The estimator

Factor models usually assume that an asset's exposures to common risk factors are
constant. In practice, betas, factor premia and idiosyncratic risk shift with
market conditions. RWLS estimates a separate factor model for every asset in every
latent market regime, without fitting a regime-switching model asset by asset.

Estimation proceeds in two stages:

1. **Regime identification.** A multivariate Gaussian hidden Markov model is fitted
   to a small set of common factor returns `f_t`. Its posterior probabilities
   `γ_t(s) = P(S_t = s | data)` measure how strongly each period belongs to each
   regime `s`.

2. **Regime-weighted regressions.** For each asset `i` and regime `s`, the
   loadings solve a weighted least-squares problem in which each observation is
   weighted by its posterior regime probability:

   ```
   (α_is, β_is) = argmin  Σ_t γ_t(s) · (r_it − α − β′ f_t)²
   ```

   The regime-specific residual variance is measured on the regime's effective
   sample, the posterior mass `Σ_t γ_t(s)`:

   ```
   σ²_is = Σ_t γ_t(s) · e_it² / (Σ_t γ_t(s) − k)
   ```

   where `k` is the number of regression coefficients.

Every observation contributes to every regime in proportion to the evidence that
it belongs there. Assets are not hard-classified, and no observation is discarded.
Because all assets share the same factor history and weights, the regressions
for a whole cross-section are solved jointly. The cost of adding an asset is one
weighted regression per regime, which makes the approach practical for universes
of thousands of securities.

## From loadings to regime-conditional moments

Combined with the regime-conditional factor means `λ_s` and covariances `Ω_s`
from the hidden Markov model, the loadings imply asset-level moments in each
regime:

```
μ_s = α_s + B_s λ_s
Σ_s = B_s Ω_s B_s′ + diag(σ²_s)
```

Weighting these by forecast regime probabilities gives the asset return
distribution as a Gaussian mixture. Its mean and covariance are available in
closed form.

## Planned scope

- Hidden Markov regime identification on factor returns, with state alignment
  across refits and one-step regime forecasts.
- Vectorized RWLS estimation for large cross-sections, with effective sample
  sizes and sampling covariances of the regime-specific loadings.
- Regime-conditional and mixture moments of asset returns.
- Simulation tools for regime-switching factor models, for Monte Carlo
  evaluation of the estimators.

Applications include conditional beta estimation, regime-dependent risk models
and portfolio construction. Portfolio optimization is one use of the estimated
moments, not the purpose of the package.

## Status

Under development. This release reserves the package name; the implementation
will follow in a later version.

## License

MIT
