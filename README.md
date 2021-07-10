# Jchemo.jl

## Dimension reduction, Regression and Discrimination for Chemometrics
## <span style="color:grey70"> Version 0.0-0 </span> 
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

Huge variety of pipelines exist in chemometrics and machine learning. Only few examples are provided in the package. The user can build some of his own pipelines with the provided elementary functions. 

Generic functions such as **transform**, **predict**, **coef** and **summary** are available. The tuning of the prediction models is facilitated by functions **gridscore** (validation dataset) and **gridcv** (cross-validation), with specific fast versions for models based on latent variables (LVs) and ridge regularization.

An example of fitting and description of a partial least squares regression model is given below:

```julia
n = 6 ; p = 7 ; q = 2 ; m = 3 ;
Xtrain = rand(n, p) ; Ytrain = rand(n, q) ;
Xtest = rand(m, p) ; Ytest = rand(m, q) ;

nlv = 5 ; 
fm = plskern(Xtrain, Ytrain; nlv = nlv)

summary(fm, Xtrain).explvar

transform(fm, Xtest)
transform(fm, Xtest; nlv = 1)

coef(fm)
coef(fm; nlv = 2)

predict(fm, Xtest).pred
predict(fm, Xtest; nlv = 0:3).pred 

pred = predict(fm, Xtest).pred ;
msep(pred, Ytest)

gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = plskern, nlv = 0:nlv)
```

## <span style="color:green"> **Available functions** </span> 

Click [**HERE**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/functions.md) to see the **list of the available functions**, and [**HERE**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/demos/ex/) to see **examples**.

Main of the examples given in the the files **.jl** are fictive, built only for illustrating the syntax. The user can replace the simulated fictive data by its own datasets.

Most of the functions have a help page, e.g.

```julia
?savgol
```

## <span style="color:green"> **News** </span> 

Click [**HERE**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/news.md) to see **what changed** in the last version. 

## <span style="color:green"> **Dependent packages** </span> 

**Jchemo** is dependent to the following packages:

| Package |
|---|
| LinearAlgebra |
| Statistics |
| Distributions |
| ImageFiltering | 
| Distances | 
| NearestNeighbors | 

## <span style="color:green"> **Installation** </span> 

In order to install Jchemo, run

```julia
pkg> add https://github.com/mlesnoff/Jchemo.jl.git
```

## <span style="color:green"> **Usage** </span>

Run
```julia
using Jchemo
```

## <span style="color:green"> **Author** </span> 

**Matthieu Lesnoff**

- Cirad, [**UMR Selmet**](https://umr-selmet.cirad.fr/en), Montpellier, France

- [**ChemHouse**](https://www.chemproject.org/ChemHouse), Montpellier

**matthieu.lesnoff@cirad.fr**

### How to cite

Lesnoff, M. 2021. Julia package Jchemo: Dimension reduction, Regression and Discrimination for Chemometrics. https://github.com/mlesnoff/Jchemo. CIRAD, UMR SELMET, Montpellier, France





