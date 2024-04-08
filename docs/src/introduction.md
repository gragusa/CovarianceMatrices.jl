# Introduction

CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl/) implements several covariance estimators for stochastic processes. Using these covariances as building blocks, the package extends [GLM.jl](https://github.com/gragusa/CovarianceMatrices.jl/) to permit inference about the GLM's coefficients in those cases where the distributions of the random variables are not independently and identically distributed.

Three classes of estimators are considered:

1. **HAC** 
    - heteroskedasticity and autocorrelation consistent (Andrews, 1996; Newey and West, 1994)
2. **VARHAC**    
3. **HC**  
    - heteroskedasticity consistent (White, 1982)
4. **CRVE** 
    - cluster robust (Arellano, 1986)

# Long run covariance

For what follows, let $\{X_t, t\in\mathBB{N}\}$ be a stochastic process, i.e., a sequence of random vectors (r.v.). We will assume throughout that the r.v. are $p$-dimensional. The sample average of the process is defined as
$$
\bar{X}_T = \frac{1}{n} \sum_{t=1}^n X_t 
$$
It is often the case that the sampling distribution of $\sqrt{T}\bar{X}_T$ can be approximated (as $n\to\infty$) by the distribution of a standard multivariate normal distribution centered at $\mu_T:=E(\bar{X}_T)$ and with *asymptotic variance-covariance* $V$:
$$
\sqrt{n}V^{-1/2}(\bar{X}_T - \mu_T) \xrightarrow{d} N(0, I_{p}),
$$
where 
$$
V_T \equiv \lim_{n\to\infty} \frac{1}{T} E\left[(\sum_{t=1}^T X_t - \bar{X}_T)(\sum_{t=1}^T X_t- \bar{X}_T)'\right].
$$
Estimation of $V$ is central in many statistics applications. For instance, we might be interested in constructing asymptotically valid confidence intervals for a linear combination of the unknown expected value of the process $\mu_T$; that is, we are interested in carrying out inference about $c'\mu_T$ for some $p$-dimensional vector $c$. For any random random matrix $\hat{V}_T$ tending in probability (as $n\to\infty$) to $V_T$, a confidence interval for $c'\bar{X}_T$ with asymptotic coverage $(\alpha\times 100)\%$ is given by
$$
\left[c'\bar{X}_{T}-q_{(1-\alpha)/2}\frac{c'\hat{V}_Tc}{\sqrt{T}},c'\bar{X}_{T}+q_{(1-\alpha)/2}\frac{c'\hat{V}_Tc}{\sqrt{T}}\right]
$$
where $q_{\alpha}$ is the $\alpha$-quantile of the standard normal distribution: $\Pr(N(0,1)\leqslant q_{\alpha}) = \alpha$.

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
