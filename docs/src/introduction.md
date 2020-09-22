# Introduction

CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl/) implements several covariance estimator for stochastic process. Using these covariances as building blocks, the package extends [GLM.jl](https://github.com/gragusa/CovarianceMatrices.jl/) to alllow obtaining robust covariance estimates of GLM's coefficients. 

Three classes of estimators are considered:

1. **HAC** 
    - heteroskedasticity and autocorrelation consistent (Andrews, 1996; Newey and West, 1994)
2. **VARHAC**    
3. **HC**  
    - heteroskedasticity consistent (White, 1982)
4. **CRVE** 
    - cluster robust (Arellano, 1986)

# Long run covariance

For what follows, let ``\{X_t\}`` be a stochastic process, i.e. a sequence of random vectors. We will assume throughout that the r.v. are $p$-dimensional. The sample average of the process is defined as
```math
\bar{X}_n = \frac{1}{n} \sum_{t=1}^n X_t 
```
It is often the case that the sampling distribution of ``\sqrt{n}\bar{X}_n`` can be approximated (as ``n\to\infty``) by the distribution of a standard multivariate normal distribution centered at ``\mu`` and *long-run variance-covariance* $V$. In other words, the following holds
```math
\sqrt{n}V^{-1/2}(\bar{X}_n - \mu_n) \xrightarrow{d} N(0, I_{p}),
```
where 
```math 
\mu = E(X_t), \quad \text{and} \quad V \equiv \lim_{n\to\infty} 
```

Estimation of $V$ is central in many applications of statistics. For instance, we might be interested in constructing asymptotically valid conficence intervals for a linear combination of the unknown expected value of the process $\mu$, that is, we are interested in making inference about ``c'\mu`` for some ``p``-dimensional vector ``c``. For any random random matrix ``\hat{V}`` tending in probability (as ``n\to\infty``) to $V$, a confidence interval for ``c'\bar{X}`` with asymptotic coverage ``(\alpha\times 100)\%`` is given by
```math
\left[c'\bar{X}_{n}-q_{(1-\alpha)/2}\frac{c'\hat{V}c}{\sqrt{n}},c'\bar{X}_{n}+q_{(1-\alpha)/2}\frac{c'\hat{V}c}{\sqrt{n}}\right]
```
where ``q_{\alpha}`` is the ``\alpha``-quantile of the standard normal distribution.

[CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl/) provides methods to estimate $V$ under a variety of assumption on the correlation stracture of the random process. We know explore them one by one starting from the simplest case. 

## Serially uncorrelated process

If the process is uncorrelated, the variance covariance reduces to
```math
V_{n}=E\left[\frac{1}{n}\sum_{t=1}^{n}(X_{t}-\mu)(X_{t}-\mu)'\right]
```
An consistent estimator of $V$ is thus given by
``math
\hat{V}_{n}=\frac{1}{n}\sum_{t=1}^{n}(X_{t}-\bar{X}_{n})(X_{t}-\bar{X}_{n})'.
``
Given `X::AbstractMatrix` with `size(X)=>(n,p)` containing ``n`` observations on the ``p`` dimensional random vectors, an estimate of $V$ can be obtained by `lrvar`:
```julia
Vhat = lrvar(Uncorrelated(), X)
```
`Uncorrelated` is the type signalling that the random sequence is assumed to be uncorrelated. 

# Api

## Serially correlated process



## Correlated process (time-series)

## Correlated process (spatial)

In this case, 