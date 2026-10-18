# rwls

Regime-weighted least squares (RWLS) for estimating regime-dependent factor models.

## The estimator

Factor models usually assume that an asset's exposures to common risk factors are
constant. In practice, betas, factor premia and idiosyncratic risk shift with
market conditions. RWLS estimates a separate factor model for every asset in every
latent market regime, without fitting a regime-switching model asset by asset.

Estimation proceeds in two stages.

1. Regime identification. A multivariate Gaussian hidden Markov model is fitted to
   a small set of common factor returns `f_t`. Its posterior probabilities
   `γ_t(s) = P(S_t = s | data)` measure how strongly each period belongs to each
   regime `s`.

2. Regime-weighted regressions. For each asset `i` and regime `s`, the loadings
   solve a weighted least-squares problem in which each observation is weighted
   by its posterior regime probability:

   ```
   (α_is, β_is) = argmin  Σ_t γ_t(s) * (r_it - α - β' f_t)^2
   ```

   The regime-specific residual variance is measured on the regime's effective
   sample, the posterior mass `n_s = Σ_t γ_t(s)`:

   ```
   σ²_is = Σ_t γ_t(s) * e_it^2 / (n_s - k)
   ```

   where `k` is the number of regression coefficients.

Every observation contributes to every regime in proportion to the evidence that
it belongs there. Assets are not hard-classified, and no observation is discarded.
Because all assets share the same factor history and weights, the regressions
for a whole cross-section are solved jointly. The cost of adding an asset is one
weighted regression per regime, which makes the approach practical for universes
of thousands of securities.

## From loadings to regime-conditional moments

The factor means `m_s` and covariances `V_s` of each regime are computed with the
same posterior weights as the regressions. Together with the loadings they imply
asset-level moments in each regime:

```
μ_s = α_s + B_s m_s
Σ_s = B_s V_s B_s' + diag(σ²_s)
```

Weighting these by forecast regime probabilities gives the distribution of next
period's asset returns as a Gaussian mixture, whose mean and covariance are
available in closed form.

The full specification, including every estimation choice and default, is in
[docs/methodology.md](docs/methodology.md).

## Planned scope

- Factor data: user-supplied factor returns, or a built-in loader for the Kenneth
  R. French Data Library (CAPM, Fama-French three and five factors, with or
  without momentum) that records the data vintage.
- Regime identification: hidden Markov model fitting with warm-started refits,
  posterior and forecast regime probabilities, alignment and sorting of regime
  labels, and selection of the number of regimes by AIC, BIC, HQIC and ICL over
  multiple starts.
- Estimation: vectorized RWLS for large cross-sections, with posterior masses,
  sandwich standard errors, t or normal p-values, confidence intervals, R-squared
  and a Wald test of equal loadings across regimes.
- Results in the style of a regression package: text summaries, HTML display in
  notebooks, and conversion to data frames.
- Moments: regime-conditional and mixture moments of asset returns, kept in
  factor form for large universes.
- Re-estimation through time: fits date by date on the data available at each
  date, for out-of-sample work.
- Settings: every constant of the estimators exposed as a documented default.

Applications include conditional beta estimation, regime-dependent risk models
and portfolio construction. These are left to the user: the package covers the
estimator and the moments it implies.

## Status

Under development. The settings, the factor loader and the regime model are in
place; the estimator, results and moments are being added. Version 0.0.1 on PyPI
only reserves the package name. The first usable release will be 0.1.0.

## License

MIT
