# Jchemo.jl

## Dimension reduction, Regression and Discrimination for Chemometrics 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/dev)
[![Build Status](https://github.com/mlesnoff/Jchemo.jl/workflows/CI/badge.svg)](https://github.com/mlesnoff/Jchemo.jl/actions)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

**Jchemo.jl** is a [**package**](https://mlesnoff.github.io/Jchemo.jl/stable/domains/) 
for **data exploration and prediction** with focus on **high dimensional data**. 

The package was initially designed about **partial least squares regression and discrimination models** and variants, in particular locally weighted PLS models (**LWPLS**) (e.g. https://doi.org/10.1002/cem.3209).
Then, it has been expanded to many other methods for 
analyzing high dimensional data. 

The name **Jchemo** comes from the fact that the package is orientated to chemometrics, but most of the provided methods are fully **generic to other domains**. 

Functions such as **transform**, **predict**, **coef** and **summary** are available. 
**Tuning the predictive models** is facilitated by generic functions **gridscore** (validation dataset) and 
**gridcv** (cross-validation). Faster versions are also available for models based on latent variables (LVs) 
(**gridscorelv** and **gridcvlv**) and ridge regularization (**gridscorelb** and **gridcvlb**).

**Examples** demonstrating the package are available in project [**JchemoDemo**](https://github.com/mlesnoff/JchemoDemo), used for trainings. **The datasets** used in the examples come from package [**JchemoData.jl**](https://github.com/mlesnoff/JchemoData.jl).

Some of the **Jchemo** functions (in particular those using kNN selections) use multi-threading 
to speed the computations. To take advantage of this, the user has to specify his relevant number 
of threads (e.g. from the setting menu of the VsCode Julia extension and the file settings.json).

**Jchemo** uses **Makie** for plotting. To display the plots, the user has to preliminary install and load one 
of the Makie's backends (e.g. **CairoMakie**). 

Most of the functions have a **help page** (providing an example), e.g.:

```julia
?savgol
```

Before to update **Jchemo**, it is recommended to have a look on 
[**What changed**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/news.md) to avoid
problems due to eventual breaking changes. 

## <span style="color:green"> **Available functions** </span> 

- [**List of functions by domain**](https://github.com/mlesnoff/Jchemo.jl/blob/master/docs/src/domains.md)
- [**Documentation**](https://mlesnoff.github.io/Jchemo.jl/dev/) 
- [**Examples**](https://github.com/mlesnoff/JchemoDemo/) Scripts (examples) illustrating the use of some functions of Jchemo.

## <span style="color:green"> **News** </span> 

- [**What changed**](https://github.com/mlesnoff/Jchemo.jl/tree/master/docs/src/news.md) between versions 

## <span style="color:green"> **Dependent packages** </span> 

- [**List of packages**](https://github.com/mlesnoff/Jchemo.jl/blob/master/Project.toml) 

## <span style="color:green"> **Installation** </span> 

In order to install Jchemo, run:

```julia
pkg> add Jchemo
```
or for the current developing version (not 100% stable):

```julia
pkg> add https://github.com/mlesnoff/Jchemo.jl.git
```

## <span style="color:green"> **Usage** </span>

Run
```julia
using Jchemo
```

## <span style="color:green"> **Benchmark - Computation time for a PLS** </span> 
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
## PLS2
## (NB.: multi-threading is not used in plskern) 
n = 10^6 # nb. observations (samples)
p = 500  # nb. X-variables (features)
q = 10   # nb. Y-variables to predict
X = rand(n, p)
Y = rand(n, q)
nlv = 25 # nb. PLS latent variables

@time plskern(X, Y; nlv = nlv) ;
8.100469 seconds (299 allocations: 4.130 GiB, 6.58% gc time)

@time plskern!(X, Y; nlv = nlv) ;
7.232234 seconds (6.47 k allocations: 338.617 MiB, 7.39% gc time, 0.13% compilation time)
```

## <span style="color:green"> **Examples of syntax for predictive models** </span> 

### **Fitting a model**

```julia
using Jchemo

n = 150 ; p = 200 ; q = 2 ; m = 50 
Xtrain = rand(n, p) ; Ytrain = rand(n, q) 
Xtest = rand(m, p) ; Ytest = rand(m, q) 

nlv = 5 
fm = plskern(Xtrain, Ytrain; nlv = nlv) ;
pnames(fm)

summary(fm, Xtrain)

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
## In the same manner, using gridscorelb for ridge regression models
## is much faster than using the generic function gridcv.

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
## In the same manner, using gridcvlb for ridge regression models
## is much faster than using the generic function gridcv.

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

- G. Cornu (Cirad) https://ur-forets-societes.cirad.fr/en/l-unite/l-equipe 
- L. Plagne, F. Févotte (Triscale.innov) https://www.triscale-innov.com 
- R. Vezy (Cirad) https://www.youtube.com/channel/UCxArXLI-gxlTmWGGgec5D7w 



