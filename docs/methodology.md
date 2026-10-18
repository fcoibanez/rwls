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
| $k$ | coefficients per regression, $K + 1$ with an intercept |
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
uncorrelated, since a change of regime moves all factor means together. A full
covariance matrix is available as an option; the diagonal form is the default.

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

The prior keeps the variances of a regime with little posterior mass away from
zero, which stops the regime from collapsing onto a handful of observations. The
default is $c_0 = 0.01$, and $c_0 = 0$ gives the maximum-likelihood estimates. In
simulations calibrated to US factor returns, the prior improves the
classification of the true regimes by several percentage points.

From a cold start, the means are initialized by $k$-means with ten restarts, the
variances of every regime by the full-sample variances plus $10^{-3}$, and the
rows of the transition matrix by draws from a Dirichlet distribution. All random
draws use a fixed seed.

The likelihood has many local maxima, and a single cold start can stop at a poor
one. By default the model is therefore fitted from ten cold starts with
consecutive seeds, and the fit with the highest likelihood is kept. Refits that
start from an earlier fit (Section 1.3) use a single start.

Iteration stops when the log-likelihood improves by less than a tolerance
$\varepsilon$, or after 1,000 iterations. The default is $\varepsilon = 10^{-4}$.
Looser tolerances can stop a warm-started refit before it reaches the optimum.

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
every regime keeps a strictly positive probability and its logarithm is finite.

### 1.6 Regime labels

Regimes are labelled 1 to $M$. EM numbers them arbitrarily: which regime comes out
as regime 1 depends on the starting values, so the same regime can carry
different labels in two fits. Estimation never depends on the labels, since
forecast probabilities and regime-conditional moments permute together. The
labels matter only when results are compared or reported, and three conventions
fix them.

A fit can be renumbered by hand, by giving the current labels in their new order.

A fit can be aligned to a reference model. Regimes are matched by minimizing the
total Euclidean distance between their mean vectors (Hungarian algorithm), after
dividing each factor by the standard deviation of the reference means across
regimes, and take the labels of the regimes they are matched to.

A fit can be sorted by the mean of one factor, so that regime 1 has the highest
mean of that factor, or the lowest if sorted in ascending order. This gives a
numbering that does not depend on the starting values. When two regimes have
nearly the same mean on the chosen factor, alignment is the more reliable rule.

Re-estimation through time needs none of these after the first date: each refit
starts from the previous one and keeps its labels.

Regimes are not given economic names. Whether a regime is better described as a
bull market, a bear market or something else depends on the factors, the
frequency and the sample, and is left to the user.

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

The intercept can be omitted, in which case row $\tau$ of $X$ is $f_\tau'$ and
$k = K$. It is included by default.

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

The denominator is the regime's degrees of freedom, treating the posterior mass
as a number of observations. The factor variances of Section 2.5 divide by the
posterior mass itself, as the maximum-likelihood estimates do. Dividing by
$t - k$ instead,
as standard regression software does for weighted least squares, understates the
variance of a regime by roughly the regime's share of the sample. The floor of one
prevents a negative or infinite variance when the posterior mass is close to the
number of coefficients, a case that Section 2.4 normally removes first.

### 2.4 Sparse regimes

A regime with too little posterior mass does not identify its regression. If

$$
n_s \le k + 1,
$$

its weights are replaced by equal weights on every observation, $w^{s}_\tau = 1$,
and a warning is issued. The regime then takes the loadings, residual variances
and factor moments of a model without regimes, estimated on the same window.
This is a safeguard for short samples. With several hundred observations, as in
the reference application, the posterior mass of every regime stays well above
the threshold.

Under hard classification, where $n_s$ is the number of classified observations
in the window, a regime that fails the same test instead takes equal weights on
the most recent $L$ observations, whichever regime they belong to.

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

## 3. Asset return moments

### 3.1 Regime-conditional moments

Stack the estimates of regime $s$ across assets: $\hat\alpha_s \in \mathbb{R}^N$
holds the intercepts, the $N \times K$ matrix $\hat B_s$ has row $i$ equal to
$\hat\beta_{is}'$, and $\hat D_s = \mathrm{diag}(\hat\sigma^2_{1s}, \dots, \hat\sigma^2_{Ns})$.
Conditional on regime $s$, the factor model implies an asset mean vector and
covariance matrix

$$
\hat\mu_s = \hat\alpha_s + \hat B_s \hat m_s, \qquad
\hat\Sigma_s = \hat B_s \hat V_s \hat B_s' + \hat D_s,
$$

with $\hat m_s$ and $\hat V_s$ from Section 2.5. Without an intercept,
$\hat\alpha_s = 0$. $\hat\Sigma_s$ is positive definite whenever every residual
variance is positive. Computed covariance matrices are symmetrized,
$\hat\Sigma_s \leftarrow (\hat\Sigma_s + \hat\Sigma_s')/2$, to remove rounding
asymmetries.

### 3.2 The predictive mixture

Given the regime forecast $\hat\pi_{t+1}$ of Section 1.5, the distribution of next
period's asset returns is the Gaussian mixture

$$
r_{t+1} \sim \sum_s \hat\pi_{t+1}(s) \mathcal{N}\big(\hat\mu_s, \hat\Sigma_s\big).
$$

Its mean and covariance are

$$
\bar\mu = \sum_s \hat\pi_{t+1}(s) \hat\mu_s, \qquad
\bar\Sigma = \sum_s \hat\pi_{t+1}(s) \big(\hat\Sigma_s + \hat\mu_s \hat\mu_s'\big) - \bar\mu \bar\mu'.
$$

The second term of $\bar\Sigma$ adds the dispersion of the regime means to the
average within-regime covariance. A single Gaussian with these two moments
matches the mixture's mean and covariance but not its skewness or tails.

### 3.3 Known regime parameters

In simulation studies the parameters of the generating model may be supplied
instead of estimated. The posterior probabilities and the regime forecast are
then computed from the true $\pi_0$, $A$, $m_s$ and $V_s$ on the data available
at $t$, and $\hat m_s$, $\hat V_s$ are replaced by $m_s$, $V_s$. The loadings and
residual variances are still estimated by Section 2, with the posterior
probabilities of the true model as weights. Comparing this variant with full
estimation separates the error from identifying regimes from the error in
everything else.

### 3.4 Factor form

The $N \times N$ matrices of Section 3.1 never need to be formed explicitly. Each
regime is fully described by $(\hat\alpha_s, \hat B_s, \hat m_s, \hat V_s, \hat D_s)$,
which takes $O(NK)$ memory instead of $O(N^2)$. Quadratic forms can be computed
from that representation directly. For a vector of weights $w$,

$$
w' \hat\Sigma_s w = (\hat B_s' w)' \hat V_s (\hat B_s' w) + \sum_i w_i^2 \hat\sigma^2_{is}.
$$

The package keeps the factor form and builds dense matrices only when they are
asked for.
