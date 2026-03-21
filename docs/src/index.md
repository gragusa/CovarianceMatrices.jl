# CovarianceMatrices.jl

*A Julia package for robust covariance matrix estimation*

```@meta
CurrentModule = CovarianceMatrices
```

## Overview

`CovarianceMatrices.jl` provides a comprehensive suite of robust covariance matrix estimators for econometric estimators. 

## Installation

```julia
using Pkg
Pkg.add("CovarianceMatrices")
```


## Main Functions

```@docs
aVar
vcov
stderror
```


## Citation

If you use CovarianceMatrices.jl in your research, please cite:

```bibtex
@misc{CovarianceMatricesJl,
  title = {CovarianceMatrices.jl: Robust Covariance Matrix Estimation for Julia},
  author = {Giuseppe Ragusa and contributors},
  year = {2024},
  url = {https://github.com/gragusa/CovarianceMatrices.jl}
}
```
