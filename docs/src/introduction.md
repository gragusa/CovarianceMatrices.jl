# Introduction

[CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl/) implements several covariance estimators for stochastic processes. Using these covariances as building blocks, the package extends [GLM.jl](https://github.com/gragusa/CovarianceMatrices.jl/) to permit inference about the GLM's coefficients in those cases where the distributions of the random variables are not independently and identically distributed.

## Long run covariance

For what follows, let $\{X_t, t\in\mathbb{Z}_{+}\}$ be a stochastic process, i.e., a sequence of random vectors (r.v.). We will assume throughout that the r.v. are $p$-dimensional. The sample average of the process is defined as
$$
\bar{X}_n = \frac{1}{n} \sum_{t=1}^n X_t 
$$
It is often the case that the sampling distribution of $\sqrt{n}\bar{X}_n$ can be approximated (as $n\to\infty$) by the distribution of a standard multivariate normal distribution centered at $\mu_n:=E(\bar{X}_n)$ and with *asymptotic variance-covariance* $V$:
$$
\sqrt{n}\Omega^{-1/2}(\bar{X}_n - \mu_n) \xrightarrow{d} N(0, I_{p}),
$$
where 
$$
\Omega_n \equiv \lim_{n\to\infty} \mathrm{Var}\left(\frac{1}{\sqrt{n}} \bar{X}_n\right)= E\left[\left(\frac{1}{\sqrt n}\sum_{t=1}^n (X_t - \bar{X}_n)\right)\left(\frac{1}{\sqrt n}\sum_{t=1}^n (X_t- \bar{X}_n)\right)'\right].
$$
Estimation of $\Omega_n$ is central in many statistics applications. For instance, we might be interested in constructing asymptotically valid confidence intervals for a linear combination of the unknown expected value of the process $\mu_n$; that is, we are interested in carrying out inference about $c'\mu_n$ for some $p$-dimensional vector $c$. For any random random matrix $\hat{\Omega}_n$ tending in probability (as $n\to\infty$) to $\Omega_n$, a confidence interval for $c'\bar{X}_n$ with asymptotic coverage $(\alpha\times 100)\%$ is given by
$$
\left[c'\bar{X}_{n}-q_{(1-\alpha)/2}\frac{c'\hat{\Omega}_nc}{\sqrt{n}},c'\bar{X}_{n}+q_{(1-\alpha)/2}\frac{c'\hat{\Omega}_nc}{\sqrt{n}}\right]
$$
where $q_{\alpha}$ is the $\alpha$-quantile of the standard normal distribution: $\Pr(N(0,1)\leqslant q_{\alpha}) = \alpha$.

## Estimation

It is often the case that we are interested in the asymptotic distribution of the estimator $\hat{\theta}_n := \theta(X_1,\ldots,X_n)$ of some parameter $\theta_0\in\mathbb{R}^d$. If show that $\hat{\theta}_n$ is asymptotically normal with asymptotic variance-covariance matrix $\Sigma_n$, i.e.,
$$
\sqrt{n}\Sigma_n^{-1/2}(\hat{\theta}_n - \theta_0) \xrightarrow{d} N(0, I_{d}),
$$
then we can use $\hat{\Sigma}_n$ consistent estimator of $\Sigma_n$ to construct asymptotically valid confidence intervals for linear combinations of the components of $\theta_0$.

::: Definition
The **asymptotic variance-covariance matrix** of an estimator $\hat{\theta}_n$ is defined as
$$
\mathrm{aVar}(\hat{\theta}_n) := \frac{\Sigma_n}{n}.
$$
:::

### Example: Ordinary Least Squares
Consider the linear regression model
$$
Y_t = X_t'\beta_0 + u_t, \quad t=1,\ldots,n,
$$
where $Y_t$ is the response variable, $X_t$ is the vector of predictors, $\beta_0\in\mathbb{R}^p$ is the parameter vector, and $u_t$ is the error term.
The Ordinary Least Squares (OLS) estimator of $\beta_0$ is given by
$$
\hat{\beta}_0 = \left(\frac{1}{n}\sum_{t=1}^n X_tX_t'\right)^{-1}\frac{1}{n}\sum_{t=1}^n X_tY_t,
$$
Under regularity conditions, the OLS estimator is asymptotically normal:

### Generalized Linear Models

GLMs are an extension of traditional linear regression models and provide a unified framework for dealing with a variety of different types of response variables (e.g., binary, count, continuous) in a consistent manner. They are widely used in statistics for modeling and analyzing data where the outcome variable does not necessarily have a normal distribution.

The basic GLM model
$$
g(\mu_i) = \eta_i, \quad i = 1,\ldots,n,
$$
where $g(\cdot)$ is the link function, $\mu_i = E(Y_i|X_i)$ and $\eta_i=X_i\beta$. The vector of coefficient is $\beta$, $Y_i$ is the response, and $X_i$ is the vector of input features.

The link function, $g(\cdot)$, provides the relationship between the linear predictor and the mean of the distribution function. It transforms the expected value of the response variable to the linear predictor. The choice of the link function depends on the distribution of the response variable. The equation is given by:

### Examples of GLMs

1. **Linear Regression**: The link function is the identity function $g(\mu) = \mu$.

2. **Logistic Regression**: Used for binary response variables. It has a binomial distribution and uses the logit link function $g(\mu_i) = \log\left(\frac{\mu_i}{1-\mu_i}\right)$.

3. **Poisson Regression**: Used for count data that follow a Poisson distribution. The link function is the natural logarithm $g(\mu) = \log(\mu)$.


The parameters of GLMs are usually estimated using Maximum Likelihood Estimation (MLE). 


The general form of the log-likelihood for a GLM can be described based on the exponential family of distributions to which most GLM distributions belong.

The exponential family of distributions can be characterized by the following probability density function (pdf) or probability mass function (pmf), for discrete distributions:

$$
f(y; \theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)
$$
where:
- $y$ is the response variable.
- $\theta$ is the natural parameter of the distribution (related to the mean).
- $\phi$ is the dispersion parameter (in some GLM formulations, $\phi$ is assumed to be known).
- $a(\phi)$, $b(\theta)$, and $c(y, \phi)$ are known functions that define the distribution.

### Log-Likelihood for GLM

Given a dataset $\{(y_i, x_i)\}_{i=1}^n$ of $n$ observations, the log-likelihood ($\mathcal{L}$) of the GLM can be expressed as:

$$
\log \mathcal{L}(\theta, \phi; y) = \sum_{i=1}^n \log f(y_i; \theta_i, \phi)
$$

Substituting the exponential family pdf/pmf, the log-likelihood becomes:

$$
\log \mathcal{L}(\theta, \phi; y) = \sum_{i=1}^n \left(\frac{y_i\theta_i - b(\theta_i)}{a(\phi)} + c(y_i, \phi)\right)
$$


Taking the derivative of the log likelihood function, set it equal
to zero, gives the score equation 
$$
\sum_{i=1}^n \frac{X_i(Y_i-\mu_i)}{b''(\theta_i)a(\phi)}\frac{\partial \mu_i}{\partial \eta_i} = 0
$$




[CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl/) provides methods to estimate $V$ under various assumptions on the correlation structure of the random process. We now explore them one by one, starting with the simplest case. 

## Serially uncorrelated process

If the process is uncorrelated, the variance-covariance reduces to
$$
V_{T}=E\left[\frac{1}{T}\sum_{t=1}^{n}(X_{t}-\mu)(X_{t}-\mu)'\right].
$$
An consistent estimator of $V_T$ is thus given by
$$
\hat{V}_{T}=\frac{1}{T}\sum_{t=1}^{n}(X_{t}-\bar{X}_{T})(X_{t}-\bar{X}_{T})'.
$$
Given `X::AbstractMatrix` with `size(X)=>(T,p)` containing ``T`` observations on the ``p`` dimensional random vectors, an estimate of $V_T$ can be obtained by:
```julia
Vhat = aVar(Uncorrelated(), X; demean = true)
```
`Uncorrelated` is the type that signals that the random sequence is assumed to be uncorrelated. 

## Serially correlated process

1## Correlated process (time-series)

## Correlated process (spatial)

In this case, 
