# Methodology

This document describes the computations in the package: the model, the
estimators, and the conventions the implementation follows where the model leaves
a choice open. The code and the tests follow this document, so any change to one
should be reflected in the other.

Each section closes with review notes on choices that are not dictated by the
method and still need a decision.

## Notation

| Symbol | Meaning |
|---|---|
| $t = 1, \dots, T$ | time periods (months in the reference application) |
| $f_t \in \mathbb{R}^K$ | returns of the $K$ common factors in period $t$ |
| $r_{it}$ | excess return of asset $i = 1, \dots, N$ in period $t$ |
| $S_t \in \lbrace 1, \dots, M \rbrace$ | latent regime in period $t$ |
| $A = (a_{sr})$ | regime transition matrix, $a_{sr} = P(S_{t+1} = r \mid S_t = s)$ |
| $m_s,   V_s$ | factor mean vector and covariance matrix in regime $s$ |
| $\gamma_t(s)$ | posterior probability of regime $s$ in period $t$ |
| $\alpha_{is},   \beta_{is}$ | intercept and factor loadings of asset $i$ in regime $s$ |
| $\sigma^2_{is}$ | residual variance of asset $i$ in regime $s$ |
| $k = K + 1$ | coefficients per regression (intercept plus loadings) |

Vectors are columns. $\mathrm{diag}(v)$ is the diagonal matrix with $v$ on
the diagonal.

## 1. Regime identification

### 1.1 Model

The regime follows a first-order Markov chain with transition matrix $A$ and
initial distribution $\pi_0$. Conditional on the regime, factor returns are
Gaussian with a diagonal covariance matrix:

$$
f_t \mid S_t = s  \sim  \mathcal{N}\big(m_s,  V_s\big), \qquad V_s = \mathrm{diag}(v_{s1}, \dots, v_{sK}).
$$

Factors are therefore uncorrelated within a regime. They are not unconditionally
uncorrelated, since a change of regime moves all factor means together.

### 1.2 Estimation

Parameters are estimated by expectation-maximization (Baum-Welch) on
$f_1, \dots, f_T$, using the scaled forward-backward recursions.

The initial distribution is fixed at $\pi_0 = (1/M, \dots, 1/M)$ and is not
updated. With a single sequence it is not identified, and its estimate collapses
onto whichever regime the sample starts in.

The transition matrix is updated by the usual EM step,
$a_{sr} \propto \sum_t \xi_t(s, r)$, where $\xi_t(s, r)$ is the posterior
probability of a transition from $s$ to $r$ between $t$ and $t + 1$. The means
are $m_s = \sum_t \gamma_t(s) f_t  /  \sum_t \gamma_t(s)$, and the variances
include a prior $c_0$:

$$
v_{sj} = \frac{c_0 + \sum_t \gamma_t(s) (f_{tj} - m_{sj})^2}{\sum_t \gamma_t(s)}.
$$

The reference implementation uses $c_0 = 0.01$ (see R1.1).

From a cold start, the means are initialized by $k$-means with ten restarts, the
variances of every regime by the full-sample variances plus $10^{-3}$, and the
rows of the transition matrix by draws from a Dirichlet distribution. All random
draws use a fixed seed.

Iteration stops when the log-likelihood improves by less than a tolerance
$\varepsilon$, or after 1,000 iterations. The reference implementation uses
$\varepsilon = 0.1$ (see R1.2).

### 1.3 Re-estimation through time

Out of sample, the model is re-estimated at every decision date $t$ using
$f_1, \dots, f_t$ only.

Each refit starts from the previous date's estimates. Transition probabilities
are floored at $10^{-8}$ and each row is renormalized, so that a probability
estimated at zero on one date can still be revised on the next. Starting from the
previous fit keeps each refit cheap and keeps regime labels stable along a path.

A fit is accepted only if every parameter is finite and every row of $A$ sums to
one. Otherwise the model is refitted from a cold start, and an error is raised if
that fit also fails the check.

### 1.4 Posterior probabilities

Given parameters estimated on $f_1, \dots, f_t$, the forward-backward recursions
on the same sample give

$$
\gamma_\tau(s) = P(S_\tau = s \mid f_1, \dots, f_t), \qquad \tau = 1, \dots, t.
$$

For $\tau < t$ these are smoothed probabilities within the available sample. For
$\tau = t$ the backward variable equals one, so $\gamma_t(s)$ is the filtered
probability $P(S_t = s \mid f_1, \dots, f_t)$. Neither uses data after $t$.

### 1.5 Regime forecast

The one-step-ahead regime distribution is the filtered distribution propagated
through the transition matrix:

$$
\hat\pi_{t+1}(r) = \sum_s \gamma_t(s)  a_{sr}.
$$

Each probability is floored at $10^{-12}$ and the vector is renormalized, so that
the logarithm of every probability is finite in the portfolio objective of
Section 5.

### 1.6 Regime labels

EM numbers regimes arbitrarily, which matters only when results are reported.

To compare a fitted model with a reference model, regimes are matched by
minimizing the total Euclidean distance between their mean vectors (Hungarian
algorithm), after dividing each factor by the standard deviation of the reference
means across regimes. Estimation and portfolio choice never use this matching:
forecast probabilities and regime-conditional moments permute together, so the
labels do not affect them.

Regimes are named by their factor means. A regime whose momentum mean is below
$-1\%$ per month while its market mean is positive is the Reversal regime. Of the
remaining regimes, the one with the highest market mean is Bull and the one with
the lowest negative market mean is Bear. With more regimes, any further
positive-market regimes are Neutral and any further negative-market regimes Mild
bear.

### Review notes

R1.1, variance prior. $c_0 = 0.01$ is the default of the HMM library rather than
a modelling choice, and the paper does not mention it. It adds
$0.01 / \sum_t \gamma_t(s)$ to every factor variance in regime $s$. For a regime
with 60 effective months that is about $1.7 \times 10^{-4}$, a sizeable fraction
of monthly factor variances, which are of order $10^{-4}$ to $10^{-3}$. Either
set $c_0 = 0$ for pure maximum likelihood, or keep it and document it as a
regularizing prior.

R1.2, convergence tolerance. $\varepsilon = 0.1$ on the log-likelihood is ten
times the library default. Its effect on the warm-started refits should be
measured before it is fixed.

R1.3, `min_covar`. The library's `min_covar` ($10^{-3}$ here) enters only the
cold-start initialization. It is not a floor on the estimated variances, as the
name might suggest.

R1.4, diagonal covariances. The diagonal restriction on $V_s$ is a modelling
choice the paper already lists as a limitation. The package should offer a full
covariance as an option and keep the diagonal one as the default, for
comparability with the paper.
