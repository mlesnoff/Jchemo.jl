# Jchemo.jl

## Dimension reduction, Regression and Discrimination for Chemometrics
## <span style="color:grey70"> **Version 0.0-1** </span> 
## <span style="color:green"> **NOT WORKING - UNDER CONSTRUCTION** </span> 

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/mlesnoff/Jchemo.jl.svg?branch=master)](https://travis-ci.com/mlesnoff/Jchemo.jl)
[![codecov.io](http://codecov.io/github/mlesnoff/Jchemo.jl/coverage.svg?branch=master)](http://codecov.io/github/mlesnoff/Jchemo.jl?branch=master)

**Jchemo** provides elementary functions (mainly focusing on methods of dimension reduction or regularization for high dimensional data) to build ad'hoc pipelines for predictions in chemometrics or other domains. 

Only few examples of many possible pipelines useful in chemometrics and machine learning are provided in the package.  

Generic functions such as **transform**, **predict**, **coef** and **summary** are available. The tuning of the prediction models is facilitated by functions **gridscore** (validation dataset) and **gridcv** (cross-validation), with specific fast versions for models based on latent variables (LVs) and ridge regularization.

## <span style="color:green"> **Available functions** </span> 

**Click** [**HERE**](https://github.com/mlesnoff/Jchemo/blob/master/doc/Jchemo_functions_github.md) **to see the list of the available functions** 

After the package installation, all the functions have a help page with documented examples. 

## <span style="color:green"> **News** </span> 

Click [**HERE**](https://github.com/mlesnoff/Jchemo/blob/master/inst/NEWS.md) to see **what changed** in the last version 

## <span style="color:green"> **Dependent packages** </span> 

**Jchemo** is dependent to the following packages:

| Package | Which use in Jchemo? |
|---|---|
| LinearAlgebra | ... |
| Statistics | ... |
| ImageFiltering | ... |
| Distances | ... |
| NearestNeighbors | ... |

## <span style="color:green"> **Installation** </span> 

In order to install Jchemo, run

```julia
pkg> add https://github.com/mlesnoff/Jchemo.jl.git
```

## <span style="color:green"> Usage </span>

Run
```julia
using Jchemo
```

## <span style="color:green"> **Author** </span> 

**Matthieu Lesnoff**

- Cirad, [**UMR Selmet**](https://umr-selmet.cirad.fr/en), Montpellier, France

- [**ChemHouse**](https://www.chemproject.org/ChemHouse), Montpellier

**matthieu.lesnoff@cirad.fr**

#### How to cite

Lesnoff, M. 2021. Julia package Jchemo: Dimension reduction, Regression and Discrimination for Chemometrics. https://github.com/mlesnoff/Jchemo. CIRAD, UMR SELMET, Montpellier, France







