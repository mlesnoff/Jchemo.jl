# Jchemo.jl

## Julia package for machine learning with focus on chemometrics and high-dimensional data

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/dev)
[![Build Status](https://github.com/mlesnoff/Jchemo.jl/workflows/CI/badge.svg)](https://github.com/mlesnoff/Jchemo.jl/actions)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

**Jchemo.jl** provides tools for [**exploratory data analyses and supervised predictions**](https://mlesnoff.github.io/Jchemo.jl/dev/domains/), with focus on **high dimensional data**. 

**Changes from the last version:** see [here](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/news.md).

The package was initially designed about **partial least squares regression (PLSR) and discrimination (PLSDA) models** 
and their numerous variants, in particular locally weighted PLS models (**LWPLS-R & -DA**) (e.g. https://doi.org/10.1002/cem.3209).
The package has then been expanded to many other **dimension reduction** methods, and **regression and discrimination** models (see [here](https://mlesnoff.github.io/Jchemo.jl/dev/domains/)). 

The package was named **Jchemo** since it is orientated to chemometrics, but most of the methods that are provided are **generic to other domains**. 

Auxiliary functions such as **transform**, **predict**, **coef** and **summary** are available. 
**Tuning the predictive models** is facilitated by grid-search functions **gridscore** (validation dataset) and **gridcv** (cross-validation), generic (same syntax) for all models. Fast versions of these functions 
are also available for models based on latent variables (LVs) (**gridscorelv** and **gridcvlv**) and 
ridge regularization (**gridscorelb** and **gridcvlb**).

Most of the functions of the package have a **help page** providing an example, e.g.:

```julia
?plskern
```

Other **examples** (notebooks and scripts) demonstrating the syntax of **Jchemo.jl** are available in the project [**JchemoDemo**](https://github.com/mlesnoff/JchemoDemo) that can be used for trainings. 

**The datasets** used in the examples (help pages and JchemoDemo.jl) are stored in the package [**JchemoData.jl**](https://github.com/mlesnoff/JchemoData.jl), a repository of datasets (chemometrics and others).

Some of the functions of the package (in particular those using kNN selections) use **multi-threading** 
to speed the computations. Taking advantage of this requires to specify a relevant number 
of threads (e.g. from the *Settings* menu of the VsCode Julia extension and the file *settings.json*).

**Jchemo.jl** uses **Makie.jl** for plotting. To install and load one of the Makie's backends (e.g. **CairoMakie.jl**) is required to display the plots. 

Before to update the package, it is recommended to have a look on 
[**What changed**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/news.md) to avoid
 eventual problems when the new version contains breaking changes. 

## <span style="color:green"> **Dependent packages** </span> 

- [**List of packages**](https://github.com/mlesnoff/Jchemo.jl/blob/master/Project.toml) 

## <span style="color:green"> **Installation** </span> 

In order to install Jchemo, run in the Pkg REPL:
```julia
pkg> add Jchemo
```

or for a specific version: 
```julia
pkg> add Jchemo@0.1.18
```
or for the current developing version (not 100% stable):
```julia
pkg> add https://github.com/mlesnoff/Jchemo.jl.git
```

## <span style="color:green"> **Benchmark - Computation time for a PLS with n = 1e6 observations** </span> 
```julia
julia> versioninfo()
Julia Version 1.8.5
Commit 17cfb8e65e (2023-01-08 06:45 UTC)
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
  CPU: 16 × Intel(R) Core(TM) i9-10885H CPU @ 2.40GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-13.0.1 (ORCJIT, skylake)
  Threads: 8 on 16 virtual cores
Environment:
  JULIA_EDITOR = code
  JULIA_NUM_THREADS = 8
```

```julia
using Jchemo

## PLS2 with 1e6 observations
## (NB.: multi-threading is not used in plskern) 
n = 10^6  # nb. observations (samples)
p = 500   # nb. X-variables (features)
q = 10    # nb. Y-variables to predict
X = rand(n, p)
Y = rand(n, q)
nlv = 25  # nb. PLS latent variables

@time plskern(X, Y; nlv = nlv) ;
8.100469 seconds (299 allocations: 4.130 GiB, 6.58% gc time)

@time plskern!(X, Y; nlv = nlv) ;
7.232234 seconds (6.47 k allocations: 338.617 MiB, 7.39% gc time, 0.13% compilation time)
```

## <span style="color:green"> **Examples of syntax for predictive models** </span> 

### Notebook examples on PLSR (see JchemoDemo.jl for more)

- [Model fitting](https://github.com/mlesnoff/JchemoDemo/blob/main/Examples_Jchemo/ipynb/Regression/tecator_plsr.ipynb)
- [Tuning with gridcvlv](https://github.com/mlesnoff/JchemoDemo/blob/main/Examples_Jchemo/ipynb/Regression/tecator_gridcv_plsr.ipynb)
- [Tuning with gridscorelv](https://github.com/mlesnoff/JchemoDemo/blob/main/Examples_Jchemo/ipynb/Regression/challenge2018_gridscore_plsr.ipynb)

### **Fitting a model**

```julia
using Jchemo

n = 150 ; p = 200 ; q = 2 ; m = 50 
Xtrain = rand(n, p) ; Ytrain = rand(n, q) 
Xtest = rand(m, p) ; Ytest = rand(m, q) 

## Model fitting
nlv = 5 
fm = plskern(Xtrain, Ytrain; nlv = nlv) ;
pnames(fm) # print the names of objects contained in 'fm'

## Some summary
summary(fm, Xtrain)

## Computation of the PLS scores (LVs) for Xtest
Jchemo.transform(fm, Xtest)
Jchemo.transform(fm, Xtest; nlv = 1)

## PLS b-coefficients
Jchemo.coef(fm)
Jchemo.coef(fm; nlv = 2)

## Predictions and performance of the fitted model
res = Jchemo.predict(fm, Xtest) 
res.pred
rmsep(res.pred, Ytest)
mse(res.pred, Ytest)

Jchemo.predict(fm, Xtest).pred
Jchemo.predict(fm, Xtest; nlv = 0:3).pred 
```

### **Tuning a model by grid-search** 

- ### With gridscore

```julia
using Jchemo, StatsBase, CairoMakie

ntrain = 150 ; p = 200
ntest = 80 
Xtrain = rand(ntrain, p) ; ytrain = rand(ntrain) 
Xtest = rand(ntest, p) ; ytest = rand(ntest)
## Train is splitted to Cal+Val to tune the model,
## and the generalization error is estimated on Test.
nval = 50
s = sample(1:ntrain, nval; replace = false) 
Xcal = rmrow(Xtrain, s)
ycal = rmrow(ytrain, s)
Xval = Xtrain[s, :]
yval = ytrain[s]

## Computation of the performance over the grid
## (the model is fitted on Cal, and the performance is 
## computed on Val)
nlv = 0:10 
res = gridscorelv(
    Xcal, ycal, Xval, yval;
    score = rmsep, fun = plskern, nlv = nlv) 

## Plot the results
plotgrid(res.nlv, res.y1,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

## Predictions and performance of the best model
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
fm = plskern(Xtrain, ytrain; nlv = res.nlv[u]) ;
res = Jchemo.predict(fm, Xtest) 
rmsep(res.pred, ytest)

## *Note*: For PLSR models, using gridscorelv is much faster
## than using the generic function gridscore.
## In the same manner, for ridge regression models,
## gridscorelb is much faster than gridscore.

## Syntax for the generic gridscore
pars = mpar(nlv = nlv)
res = gridscore(
    Xcal, ycal, Xval, yval;
    score = rmsep, fun = plskern, pars = pars) 
```

- ### With gridcv

```julia
using Jchemo, StatsBase, CairoMakie

ntrain = 150 ; p = 200
ntest = 80 
Xtrain = rand(ntrain, p) ; ytrain = rand(ntrain) 
Xtest = rand(ntest, p) ; ytest = rand(ntest)
## Train is used to tune the model,
## and the generalization error is estimated on Test.

## Build the cross-validation (CV) segments
## Replicated K-Fold CV
K = 5      # Nb. folds
rep = 10   # Nb. replications (rep = 1 ==> no replication)
segm = segmkf(ntrain, K; rep = rep)

## Or replicated test-set CV
m = 30     # Size of the test-set
rep = 10   # Nb. replications (rep = 1 ==> no replication)
segm = segmts(ntrain, m; rep = rep) 

## Computation of the performances over the grid
nlv = 0:10 
rescv = gridcvlv(
    Xtrain, ytrain; segm = segm,
    score = rmsep, fun = plskern, nlv = nlv) ;
pnames(rescv)
res = rescv.res

## Plot the results
plotgrid(res.nlv, res.y1,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

## Predictions and performance of the best model 
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
fm = plskern(Xtrain, ytrain; nlv = res.nlv[u]) ;
res = Jchemo.predict(fm, Xtest) 
rmsep(res.pred, ytest)

## *Note*: For PLSR models, using gridcvlv is much faster
## than using the generic function gridcv.
## In the same manner, for ridge regression models,
## gridcvlb is much faster than gridcv.

## Using the generic function gridcv:
pars = mpar(nlv = nlv)
rescv = gridcv(
    Xtrain, ytrain; segm = segm,
    score = rmsep, fun = plskern, pars = pars) ;
pnames(rescv)
res = rescv.res
```

## <span style="color:green"> **Author** </span> 

**Matthieu Lesnoff**

- Cirad, [**UMR Selmet**](https://umr-selmet.cirad.fr/en), Montpellier, France

- [**ChemHouse**](https://www.chemproject.org/ChemHouse), Montpellier

**matthieu.lesnoff@cirad.fr**

## How to cite

Lesnoff, M. 2021. Jchemo: Julia package for machine learning, with focus on 
chemometrics and high-dimensional data. https://github.com/mlesnoff/Jchemo. 
UMR SELMET, Univ Montpellier, CIRAD, INRA, Institut Agro, Montpellier, France

## Acknowledgments

- G. Cornu (Cirad) https://ur-forets-societes.cirad.fr/en/l-unite/l-equipe 
- L. Plagne, F. Févotte (Triscale.innov) https://www.triscale-innov.com 
- R. Vezy (Cirad) https://www.youtube.com/channel/UCxArXLI-gxlTmWGGgec5D7w 



