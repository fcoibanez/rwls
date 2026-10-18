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

## Installation

```
pip install rwls
```

Python 3.10 or later. The dependencies are numpy, pandas, scipy, scikit-learn,
joblib and hmmlearn.

## Usage

Factor returns can come from any source as a pandas DataFrame, or from the
Kenneth R. French Data Library through the built-in loader. Models are `"capm"`,
`"ff3"`, `"ff3+mom"`, `"ff5"` and `"ff5+mom"`; returns are monthly and in decimals.

```python
import rwls

factors = rwls.load_factors("ff5+mom", start="1963-07", end="2024-12")
X = factors.drop(columns="RF")
excess = returns.sub(factors["RF"], axis=0)  # returns: monthly stock returns
```

French revises his factors when CRSP is updated. `factors.attrs` records the CRSP
vintage and the download date, and `cache_dir=` keeps the downloaded files so
that later runs use the same vintage.

Fit the regime model, then renumber the regimes by a rule of your choice, since
EM numbers them arbitrarily:

```python
regimes = rwls.RegimeModel(n_regimes=3).fit(X).sort_by("Mkt-RF")
print(regimes.summary(periods_per_year=12))
regimes.posteriors        # smoothed regime probabilities, one column per regime
regimes.forecast()        # probabilities for the next period
```

Estimate the regime-specific factor models of every asset:

```python
res = rwls.RWLS(excess, X, regimes).fit()
print(res["AAPL"].summary())    # regression table of one asset, by regime
print(res.summary())            # loadings across assets, by regime

res.params                      # coefficients: rows (regime, term), one column per asset
res.bse, res.pvalues            # sandwich standard errors and p-values
res.conf_int()
res.rsquared                    # weighted R-squared by regime
res.wald_equal_loadings()       # test that loadings are equal in every regime
```

Assets with missing returns are estimated on the periods they are observed.

Regime-conditional and mixture moments of asset returns, in factor form:

```python
m = res.moments()
m.mean(1), m.cov(1)                         # regime 1
mean, cov = m.mixture(regimes.forecast())   # next period's mixture
m.mixture_variance(weights, regimes.forecast())
```

Choose the number of regimes by information criteria over many starts:

```python
sel = rwls.select_regimes(X, n_regimes=range(1, 7), n_starts=100, n_jobs=-1)
sel.table       # log-likelihood, AIC, BIC, HQIC, ICL, distinct optima
sel.selected    # the number of regimes each criterion selects
```

Re-estimate through time on the data available at each date, with each regime
fit starting from the previous one:

```python
for step in rwls.expanding(excess, X, start=600):
    step.date, step.forecast, step.results.moments()
```

Every constant of the estimators is a setting with a documented default:

```python
rwls.RegimeSettings()   # number of regimes, starts, tolerance, variance prior, ...
rwls.RWLSSettings()     # intercept, weighting, fallback, t or normal inference, ...
rwls.RegimeModel(n_regimes=3, n_starts=1000, selection="modal")
```

With `selection="modal"`, the model kept is the one reached by the most starts
rather than the one of highest likelihood.

## Status

The API may still change before version 1.0.

## License

MIT
