# Jchemo.jl

## Dimension reduction, Regression and Discrimination for Chemometrics
## <span style="color:grey70"> Version 0.1.5 </span> 

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/dev)
[![Build Status](https://github.com/mlesnoff/Jchemo.jl/workflows/CI/badge.svg)](https://github.com/mlesnoff/Jchemo.jl/actions)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

**Jchemo** provides functions and pipelines for predictions in chemometrics or other domains, with focus
on high dimensional data. 

The package was initially about **k-nearest neighbors locally weighted partial least squares regression 
and discrimination** models (kNN-LWPLSR an kNN-LWPLSDA; e.g. Lesnoff et al 2021 https://doi.org/10.1002/cem.3209).
It has now been expanded to other methods. 

Generic functions such as **transform**, **predict**, **coef** and **summary** are available. 
Tuning the models for prediction is facilitated by functions **gridscore** (validation dataset) and 
**gridcv** (cross-validation). Faster versions are also available for models based on latent variables (LVs) 
(**gridscorelv** and **gridcvlv**) and ridge regularization (**gridscorelb** and **gridcvlb**).

Most of the functions have a **help page** (each given an example), e.g.:

```julia
?savgol
```

Additional examples of Jchemo scripts are available at [**JchemoTraining**](https://github.com/mlesnoff/JchemoTraining) 
(and some [**here**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/demos/ex/)). 
Datasets used in the examples are stored in package [**JchemoData**](https://github.com/mlesnoff/JchemoData.jl).

## <span style="color:green"> **Available functions** </span> 

- [**List of functions by domain**](https://github.com/mlesnoff/Jchemo.jl/blob/master/docs/src/domains.md)
- [**Documentation**](https://mlesnoff.github.io/Jchemo.jl/dev/) 

## <span style="color:green"> **News** </span> 

- [**What changed**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/news.md) between versions 

## <span style="color:green"> **Dependent packages** </span> 

- [**List of packages**](https://github.com/mlesnoff/Jchemo.jl/blob/master/Project.toml) 

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

## <span style="color:green"> **Examples of syntax** </span> 

### **Fitting a model**

```julia
using Jchemo

n = 150 ; p = 200 ; q = 2 ; m = 50 
Xtrain = rand(n, p) ; Ytrain = rand(n, q) ;
Xtest = rand(m, p) ; Ytest = rand(m, q) ;

nlv = 5 
fm = plskern(Xtrain, Ytrain; nlv = nlv) ;
pnames(fm)

summary(fm, Xtrain, Ytrain)

Jchemo.transform(fm, Xtest)
Jchemo.transform(fm, Xtest; nlv = 1)

Jchemo.coef(fm)
Jchemo.coef(fm; nlv = 2)

res = Jchemo.predict(fm, Xtest) ;
res.pred
rmsep(res.pred, Ytest)
mse(res.pred, Ytest)

Jchemo.predict(fm, Xtest).pred
Jchemo.predict(fm, Xtest; nlv = 0:3).pred 
```

### **Tuning a model by grid-search** 

- ### With gridscore

```julia
using Jchemo, CairoMakie

n = 150 ; p = 200 ; m = 50 
Xtrain = rand(n, p) ; ytrain = rand(n) 
Xval = rand(m, p) ; yval = rand(m) 

nlv = 0:10 
pars = mpar(nlv = nlv)
res = gridscore(
    Xtrain, ytrain, Xval, yval;
    score = rmsep, fun = plskern, pars = pars) 

plotgrid(res.nlv, res.y1,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
fm = plskern(Xval, yval; nlv = res.nlv[u]) ;
res = Jchemo.predict(fm, Xval) ;
rmsep(res.pred, yval)

## For PLSR models, using gridscorelv is much faster than gridscore!!!

res = gridscorelv(
    Xtrain, ytrain, Xval, yval;
    score = rmsep, fun = plskern, nlv = nlv) 
```

- ### With gridcv

```julia
using Jchemo

n = 150 ; p = 200 ; m = 50 
Xtrain = rand(n, p) ; ytrain = rand(n) 
Xval = rand(m, p) ; yval = rand(m) 

segm = segmkf(n, 5; rep = 5)     # Replicated K-fold cross-validation
#segm = segmts(n, 30; rep = 5)   # Replicated test-set validation

nlv = 0:10 
pars = mpar(nlv = nlv)
zres = gridcv(
    Xtrain, ytrain; segm,
    score = rmsep, fun = plskern, pars = pars) ;
pnames(zres)
res = zres.res

plotgrid(res.nlv, res.y1,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
fm = plskern(Xval, yval; nlv = res.nlv[u]) ;
res = Jchemo.predict(fm, Xval) ;
rmsep(res.pred, yval)

## For PLSR models, using gridcvlv is much faster than gridcv!!!

zres = gridcvlv(
    Xtrain, ytrain; segm,
    score = rmsep, fun = plskern, nlv = nlv) ;
zres.res
```

## <span style="color:green"> **Author** </span> 

**Matthieu Lesnoff**

- Cirad, [**UMR Selmet**](https://umr-selmet.cirad.fr/en), Montpellier, France

- [**ChemHouse**](https://www.chemproject.org/ChemHouse), Montpellier

**matthieu.lesnoff@cirad.fr**

## How to cite

Lesnoff, M. 2021. Jchemo: a Julia package for dimension reduction, regression and discrimination for 
chemometrics. https://github.com/mlesnoff/Jchemo. CIRAD, UMR SELMET, Montpellier, France

## Acknowledgments

- L. Plagne, F. Févotte (Triscale.innov) https://www.triscale-innov.com 
- R. Vezy (Cirad) https://www.youtube.com/channel/UCxArXLI-gxlTmWGGgec5D7w 



