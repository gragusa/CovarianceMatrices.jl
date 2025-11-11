This is a guide to implement the last feature of the package. After this feature is implemented we will go bug hunting and performance improving. 

The idea is to implemente variance estimation with smoothing moments. As far as CovarianceMatrices.jl is concerne the moment is the matrix being passed to aVar. This smoothed moments offer an alternative to the HAC estimate. 

Notice that in CovarianceMatrices.jl there is an half broken implementation (let's say broken!) of it. But kernel and smoothing function can be reused. There is to glue everyting together and make it coherent with the API. The implemenation is in smoothing.jl. I repeat i do not like the implementation, so free to change it. The only thing is that I would need a robust smoother for matrix to be in place to be used by other packages. Bot smoothing in place and smoothing out-of-place (even if in
this package we will only use the out-of-place version. Also, the smoothing should be very efficient and use loops (as done in the existing code). 

The plan is:

1) review the existing code
2) the description of the method below,
3) draft a plan to make this work with the current API
4) implement the plan
5) Build test (unfortunately there is no implementation we can use to double check the results. What we can do is to simulate data for n very large and compare it with the HAC implementation. They should get very near asymptotically. In the test use the StableRNGs RNG so we now tests are stabel wrt to change to the random number generator. 


Below a description of the method:






# What is being estimated?

You have moment conditions $g_t(\beta)\in\mathbb{R}^m$ (rows of your $n\times m$ matrix, one row per $t$). The long-run variance (LRV) you want is

$$
\Omega = \sum_{s=-\infty}^{\infty}\Gamma(s),\qquad \Gamma(s)=\mathbb{E}[g{t+s}(\beta_0)g_t(\beta_0)].
$$

Smith’s trick is to **smooth the moments themselves** first and then take a simple outer-product—this yields an **automatically p.s.d. HAC** estimator.&#x20;

# Smoothed moments

Define the smoothed moment at each time $t$ as a kernel-weighted average of nearby moments:

$$
g^{T}_t(\beta)\;=\;\frac{1}{S_T}\sum_{s=t-T}^{t-1} k\!\left(\frac{s}{S_T}\right)\, g_{t-s}(\beta),
\qquad t=1,\dots,T.
$$

Here $S_T$ is a bandwidth and $k(\cdot)$ is a kernel function (defined below). This is Eq. (2.2) in Smith (2011) and the same construction in Smith (2005). &#x20;

# The variance estimator (two equivalent normalizations)

Let $k_j=\int_{-\infty}^{\infty} k(a)^j\,da$. Then a clean, **consistent** and p.s.d. estimator of $\Omega$ is

$$
\widehat{\Omega}_T(\hat\beta)\;=\;\frac{S_T}{T\,k_2}\sum_{t=1}^T g^{T}_t(\hat\beta)\, g^{T}_t(\hat\beta)'. \tag{A}
$$

This is Eq. (2.13) in Smith (2011).&#x20;

Smith (2005) uses an equivalent **discrete** normalization:

$$
\widehat{\Omega}_T(\hat\beta)\;=\;\frac{1}{\sum_{s=-(T-1)}^{T-1} k\!\left(\frac{s}{S_T}\right)^2}\;
\sum_{t=1}^T g^{T}_t(\hat\beta)\, g^{T}_t(\hat\beta)'. \tag{B}
$$

That’s his Eq. (2.5): normalized outer product of smoothed moments; it’s automatically p.s.d. Both (A) and (B) are first-order equivalent (replace the integral constant $k_2$ with the corresponding sample sum of squares, or vice-versa).&#x20;

> In code: compute $g^{T}_t$ for all $t$; stack them in a $T\times m$ matrix $G^{T}$. Then
> $\,\widehat{\Omega} = c\cdot (G^{T})' G^{T}/T$, with $c=S_T/k_2$ (continuous) or $c=1/\sum k^2$ (discrete).

# What is the kernel, exactly?

You choose a **smoothing kernel** $k(\cdot)$ (on the “observation” scale), and it induces a standard “lag” kernel $k^*(\cdot)$ via

$$
k^*(a) \;=\; \frac{1}{k_2}\int_{-\infty}^{\infty} k(b-a)\,k(b)\,db,
$$

which belongs to the usual **p.s.d. HAC kernel class $K_2$**. Equivalently, in frequency domain, $K^*(\lambda)=2\pi\,|K(\lambda)|^2/k_2$ where $K$ and $K^*$ are the spectral windows of $k$ and $k^*$. This is why Smith’s smoother yields a HAC estimator with good properties.  &#x20;

## Concrete kernel choices & bandwidth orders (handy for coding)

Smith (2011, §2.6) lists $k(\cdot)$ choices that imply standard $k^*(\cdot)$ and known bandwidth rates:

* **Truncated/Uniform $k$** (box): $k(x)=1(|x|\le 1)$ → induced $k^*$ is **Bartlett**; optimal $m_T\propto T^{1/3}$ (with $S_T=(2m_T+1)/2$).&#x20;
* **Bartlett $k$** (triangular): $k(x)=1-|x| \ (|x|\le 1)$ → induced $k^*$ is **Parzen**; optimal $m_T\propto T^{1/5}$.&#x20;
* **Quadratic Spectral (QS) induced $k^*$**: pick $K(\lambda)=\sqrt{K^*(\lambda)}$ to back out $k$; this yields the familiar QS $k^*$ and again $S_T\propto T^{1/5}$. (Smith gives closed form using Bessel $J_1$.)&#x20;

General requirements: $S_T\to\infty$, $S_T=o(T^{1/2})$; $k(\cdot)$ bounded, continuous a.e.; and an integrability condition ensuring discrete sums approximate integrals. &#x20;

# How to compute it (drop-in algorithm)

Inputs:

* $G$ = $T\times m$ matrix of moments $g_t(\hat\beta)$.
* `kernel(x)` = function implementing $k(x)$.
* `S_T` = bandwidth (integer or float; see rates above).

Steps:

1. **Build weights** for integer shifts $s\in\{-(T-1),\dots,T-1\}$:

   $$
   w_s \;=\; \frac{1}{S_T}\,k\!\left(\frac{s}{S_T}\right).
   $$
2. **Smooth the moments** (a 1-D convolution across $t$ for each column of $G$):

   $$
   g^{T}_t\;=\;\sum_{s} w_s\, g_{t-s}\quad(\text{use zero/valid padding as in Smith’s sums}).
   $$

   This yields $G^{T}\in\mathbb{R}^{T\times m}$.
3. **Normalization constant**:

   * Continuous: compute $k_2=\int k(x)^2dx$ (closed forms exist for the examples above), set $c=S_T/k_2$.&#x20;
   * Discrete: set $c = 1/\sum_{s} k(s/S_T)^2$.&#x20;
4. **Variance**:

   $$
   \widehat{\Omega} \;=\; c\cdot \frac{(G^{T})' G^{T}}{T}.
   $$

That’s it—no lag-window on autocovariances, no Toeplitz summations. You still get a bona-fide HAC (and p.s.d.) estimator because the smoothing + outer product implicitly corresponds to a standard lag kernel $k^*$.&#x20;

# Tiny code sketch (language-agnostic)

```python
# G: (T, m) matrix of moments
# kernel(x): implements k(x)
# S: bandwidth (float); use m_T = c*T**alpha if you want integer support
import numpy as np

T, m = G.shape
# build discrete weights over s = -(T-1) ... (T-1)
s = np.arange(-(T-1), T)
w = (1.0/S) * np.array([kernel(si/S) for si in s])

# smooth each column by 1-D convolution
GT = np.vstack([np.sum(G*np.roll(w, t - (T-1))[:,None], axis=0) for t in range(T)])

# normalization (choose one)
k2 = known_k2_for_kernel   # e.g., 2 for box, 2/3 for Bartlett k, 2π for QS case in Smith (2011)
c = S / k2
# or: c = 1.0 / np.sum((np.array([kernel(si/S) for si in s]))**2)

Omega_hat = c * (GT.T @ GT) / T     # (m x m)
```

# Notes on choosing $k(\cdot)$ and $S_T$

* If you want something simple and robust: **box $k$** (uniform) with $m_T\propto T^{1/3}$ is fine and very common. It induces Bartlett $k^*$.&#x20;
* If you want better MSE: use the **QS-induced** option with $S_T\propto T^{1/5}$.&#x20;
* Smith gives the constants $k_1=\int k,\;k_2=\int k^2$ for the examples, which you can hard-code for speed.&#x20;

---

**Citations (where each piece comes from):**

* Smoothed moments definition (Eq. 2.2), CLT/LLN for $g^T$, and variance estimator $(S_T/Tk_2)\sum g^Tg^{T\prime}$.&#x20;
* Positive-semidefinite outer-product form and discrete normalization (Eq. 2.5).&#x20;
* Induced kernel $k^*(\cdot)$ and its relation to $k(\cdot)$; spectral window relation $K^*=2\pi|K|^2/k_2$. &#x20;
* Concrete kernel examples and optimal bandwidth rates. &#x20;

If you want, tell me your preferred kernel (box, Bartlett, QS) and I’ll drop in the exact $k(\cdot)$, $k_2$, and a ready-to-run function for your codebase.

