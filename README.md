# Jchemo.jl

## Dimension reduction, Regression and Discrimination for Chemometrics
## <span style="color:grey70"> Version 0.0-0 </span> 
## <span style="color:green"> **NOT STABLE AND INCOMPLETE (UNDER CONSTRUCTION)** </span> 

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

Generic functions such as **transform**, **predict**, **coef** and **summary** are available. Tuning the models is facilitated by functions **gridscore** (validation dataset) and **gridcv** (cross-validation), with specific fast versions for models based on latent variables (LVs) and ridge regularization.

## <span style="color:green"> **Example: fitting a model** </span> 

```julia
using Jchemo

n = 150 ; p = 200 ; q = 2 ; m = 50 
Xtrain = rand(n, p) ; Ytrain = rand(n, q) ;
Xtest = rand(m, p) ; Ytest = rand(m, q) ;

nlv = 5 ; 
fm = plskern(Xtrain, Ytrain; nlv = nlv)

summary(fm, Xtrain).explvar

transform(fm, Xtest)
transform(fm, Xtest; nlv = 1)

coef(fm)
coef(fm; nlv = 2)

res = predict(fm, Xtest) ;
res.pred
msep(res.pred, Ytest)

predict(fm, Xtest).pred
predict(fm, Xtest; nlv = 0:3).pred 
```

## <span style="color:green"> **Example: Tuning a model by grid-search** </span> 

### **With gridscore**

```julia
using Jchemo, CairoMakie

n = 150 ; p = 200 ; m = 50 
Xtrain = rand(n, p) ; ytrain = rand(n) 
Xval = rand(m, p) ; yval = rand(m) 

nlv = 0:10 
pars = mpars(nlv = nlv)
res = gridscore(
    Xtrain, ytrain, Xval, yval;
    score = rmsep, fun = plskern, pars = pars) 

lines(res.nlv, res.y1,
    axis = (xlabel = "Nb. LVs", ylabel = "RMSEP"))

u = findall(isapprox.(res.y1, minimum(res.y1)))[1] ;
res[u, :]
fm = plskern(Xval, yval; nlv = res.nlv[u]) ;
res = Jchemo.predict(fm, Xval) ;
rmsep(res.pred, yval)

# For PLSR models, using gridscorelv is much faster!!!
res = gridscorelv(
    Xtrain, ytrain, Xval, yval;
    score = rmsep, fun = plskern, nlv = nlv) 
```

### **With gridcv**

```julia
using Jchemo, CairoMakie

n = 150 ; p = 200 ; m = 50 
Xtrain = rand(n, p) ; ytrain = rand(n) 
Xval = rand(m, p) ; yval = rand(m) 

segm = segmkf(n, 5; rep = 5)   # Replicated K-fold cross-validation
#segm = segmts(n, 30; rep = 5)   # Replicated test-set validation

nlv = 0:10 
pars = mpars(nlv = nlv)
res = gridcv(
    Xtrain, ytrain; segm,
    score = rmsep, fun = plskern, pars = pars) ;
pnames(res)
res.res

# For PLSR models, using gridcvlv is much faster!!!
res = gridcvlv(
    Xtrain, ytrain; segm,
    score = rmsep, fun = plskern, nlv = nlv) ;
res.res
```

## <span style="color:green"> **Available functions** </span> 

- [**List of functions**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/functions.md) 
- [**Examples**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/demos/ex/)

Main of the examples given in the the files **.jl** are fictive and built only for illustrating the syntax. The user can replace the simulated fictive data by its own datasets.

Most of the functions have a help page, e.g.

```julia
?savgol
```

## <span style="color:green"> **News** </span> 

[**What changed**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/news.md) 

## <span style="color:green"> **Dependent packages** </span> 

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

Lesnoff, M. 2021. Julia package Jchemo: Dimension reduction, regression and discrimination for chemometrics. https://github.com/mlesnoff/Jchemo. CIRAD, UMR SELMET, Montpellier, France





