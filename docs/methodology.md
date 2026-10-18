# Methodology

This document describes the computations in the package: the model, the
estimators, and the conventions the implementation follows where the model leaves
a choice open. The code and the tests follow this document, so any change to one
should be reflected in the other.

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
| $n_s$ | posterior mass of regime $s$, its expected number of observations |
| $L$ | window length for hard classification and for the sparse-regime fallback |

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

The reference implementation uses $c_0 = 0.01$.

From a cold start, the means are initialized by $k$-means with ten restarts, the
variances of every regime by the full-sample variances plus $10^{-3}$, and the
rows of the transition matrix by draws from a Dirichlet distribution. All random
draws use a fixed seed.

Iteration stops when the log-likelihood improves by less than a tolerance
$\varepsilon$, or after 1,000 iterations. The reference implementation uses
$\varepsilon = 0.1$.

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

## 2. Regime-weighted least squares

### 2.1 Observation weights

At decision date $t$, each regime $s$ receives a weight for every observation
$\tau = 1, \dots, t$. Regime-weighted least squares uses the posterior
probabilities of Section 1.4:

$$
w^{s}_\tau = \gamma_\tau(s).
$$

Every observation enters every regime's regression, in proportion to the
probability that it belongs to that regime. The total weight of regime $s$,

$$
n_s = \sum_{\tau=1}^{t} \gamma_\tau(s),
$$

is its posterior mass, the number of observations the regime accounts for in
expectation.

The package also provides hard classification, which is the rule of Costa and
Kwon (2020) and serves as a benchmark. Each observation is assigned to its most
probable regime, $\hat S_\tau = \arg\max_s \gamma_\tau(s)$, and regime $s$ keeps
only the most recent $L$ observations assigned to it:

$$
w^{s}_\tau = 1 \text{ if } \hat S_\tau = s \text{ and } \tau \text{ is among the last } L \text{ such periods}, \qquad w^{s}_\tau = 0 \text{ otherwise}.
$$

The reference implementation uses $L = 24$. With two regimes, the most probable
regime is the one with probability above one half, as in Costa and Kwon.

### 2.2 Loadings

Let $X$ be the $t \times k$ design matrix whose row $\tau$ is $(1, f_\tau')$, let
$R$ be the $t \times N$ matrix of asset excess returns, and let
$W_s = \mathrm{diag}(w^{s}_1, \dots, w^{s}_t)$. For each regime, the
intercepts and loadings of all assets solve the weighted normal equations

$$
\hat\Theta_s = \big(X' W_s X\big)^{-1} X' W_s R,
$$

where column $i$ of the $k \times N$ matrix $\hat\Theta_s$ is
$(\hat\alpha_{is}, \hat\beta_{is}')'$. Every asset shares the same design matrix
and weights, so $X' W_s X$ is formed and factorized once per regime, whatever the
number of assets. This is what makes the estimator practical for large
cross-sections.

Solving for all assets at once requires them to be observed over the same
window. An asset with a shorter history needs its own design matrix and weights,
restricted to the periods it is observed.

### 2.3 Residual variances

With residuals $e_{i\tau s} = r_{i\tau} - \hat\alpha_{is} - \hat\beta_{is}' f_\tau$,
the residual variance of asset $i$ in regime $s$ is measured on the regime's
posterior mass:

$$
\hat\sigma^2_{is} = \frac{\sum_\tau w^{s}_\tau   e^2_{i\tau s}}{\max(n_s - k, 1)}.
$$

The denominator is the regime's degrees of freedom. Dividing by $t - k$ instead,
as standard regression software does for weighted least squares, understates the
variance of a regime by roughly the regime's share of the sample. The floor of one
prevents a negative or infinite variance when the posterior mass is close to the
number of coefficients, a case that Section 2.4 normally removes first.

### 2.4 Sparse regimes

A regime with too little posterior mass does not identify its regression. If

$$
n_s \le k + 1,
$$

its weights are replaced by equal weights on the most recent $L$ observations,
$w^{s}_\tau = 1$ for $\tau > t - L$ and zero otherwise, and the loadings,
residual variances and factor moments of that regime are computed from those
weights.

### 2.5 Regime-conditional factor moments

The factor mean and covariance of each regime are computed from the same weights
as its regressions:

$$
\hat m_s = \frac{\sum_\tau w^{s}_\tau f_\tau}{\sum_\tau w^{s}_\tau}, \qquad
\hat V_s = \mathrm{diag}\left( \frac{\sum_\tau w^{s}_\tau (f_\tau - \hat m_s)(f_\tau - \hat m_s)'}{\sum_\tau w^{s}_\tau} \right).
$$

With posterior weights these are the maximum-likelihood updates of Section 1.2
without the variance prior. With hard-classification weights they are the sample
moments of the classified window. When the parameters of the generating model
are known, as in a simulation, $m_s$ and $V_s$ can be used instead.

### 2.6 Sampling covariance of the loadings

Posterior probabilities are not inverse variances, so the usual weighted
least-squares standard errors do not apply. For asset $i$, the covariance of the
coefficient estimates of regimes $s$ and $r$ is estimated by the sandwich formula

$$
\widehat{\mathrm{Cov}}\big(\hat\theta_{is}, \hat\theta_{ir}\big)
= \big(X' W_s X\big)^{-1}
\left( \sum_\tau x_\tau x_\tau'   w^{s}_\tau w^{r}_\tau   e_{i\tau s} e_{i\tau r} \right)
\big(X' W_r X\big)^{-1},
$$

where $x_\tau = (1, f_\tau')'$ and $\hat\theta_{is} = (\hat\alpha_{is}, \hat\beta_{is}')'$.
Estimates for different regimes are correlated because they use the same
observations.
