# Jchemo.jl

### Machine learning and chemometrics on high-dimensional data with Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/dev)
[![Build Status](https://github.com/mlesnoff/Jchemo.jl/workflows/CI/badge.svg)](https://github.com/mlesnoff/Jchemo.jl/actions)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

# <span style="color:green"> About </span> 

This package was initially dedicated to **partial least squares regression (PLSR) and discrimination (PLSDA) models** 
and their many extensions, in particular locally weighted PLS models (**LWPLS-R** & **-DA**; e.g. https://doi.org/10.1002/cem.3209).
The package has then been expanded to various **dimension reduction methods** and **regression and discrimination models** ([see the list of functions here](https://mlesnoff.github.io/Jchemo.jl/dev/domains/)). 

Why the name **Jchemo**?: Since it is orientated to **chemometrics** (in brief, the use of biometrics for chemistry), but most of the provided methods are **generic to other domains of application**. 

**Warning:** Major breaking changes were made between **version 0.2.4** and **version 0.3.0** (to come, **work in progress** on the current main branch). See [**What changed**](https://mlesnoff.github.io/Jchemo.jl/dev/news/) for some details on the changes. Mainly, a new syntax, said **embedded**, is proposed. 

**Jchemo** is organized between **transformers** (e.g. Pca models), **predictors** (e.g. Plsr/Plsda models) and **utility functions**. For transformers and predictors, two syntaxes are allowed: the **direct syntax** (almost the same as for versions <= 2.4.0) and the **embedded syntax**. The last is intended to make easier the building of pipelines (chains) of models, and is now favored. Only this embbeded syntax is given in the **help pages** of the functions. 

# <span style="color:green"> Tips </span> 

### Model tuning

The predictive models can be **tuned** by generic (i.e. same syntax for all models) grid-search functions: **gridscore** ("test-set" validation) and **gridcv** (cross-validation). Highly accelerated versions 
of these tuning tools are available for models based on latent variables (LVs) and 
ridge regularization.

### Help and demo

Each function of **Jchemo** has a **help page** providing an example, e.g.:

```julia
julia> ?plskern
```
The **datasets** used in the examples are stored in the package [**JchemoData.jl**](https://github.com/mlesnoff/JchemoData.jl), a repository of datasets (chemometrics and other domains).

Aditionnal **examples of scripts** demonstrating the syntax of **Jchemo** are available in the training project [**JchemoDemo**](https://github.com/mlesnoff/JchemoDemo) (this project is not still updated for Jchemo versions > 2.4.0). 


### Multi-threading

Some of the functions of the package (in particular those using kNN selections) use **multi-threading** 
to speed the computations. Taking advantage of this requires to specify a relevant number 
of threads (e.g. from the *Settings* menu of the VsCode Julia extension and the file *settings.json*).

### Plotting

**Jchemo.jl** uses **Makie.jl** for plotting. To install and load one of the Makie's backends (e.g. **CairoMakie.jl**) is required to display the plots. 

### News

Before to update the package, it is recommended to have a look on 
[**What changed**](https://mlesnoff.github.io/Jchemo.jl/dev/news/) to avoid
 eventual problems when the new version contains breaking changes. 

# <span style="color:green"> Installation </span> 

In order to install Jchemo, run in the Pkg REPL:
```julia
pkg> add Jchemo
```

or for a specific version: 
```julia
pkg> add Jchemo@0.1.18
```

or for the current developing version (not stable):
```julia
pkg> add https://github.com/mlesnoff/Jchemo.jl.git
```
# <span style="color:green">  Benchmark  </span>

```julia
julia> versioninfo()
Julia Version 1.10.0
Commit 3120989f39 (2023-12-25 18:01 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
  CPU: 16 × Intel(R) Core(TM) i9-10885H CPU @ 2.40GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, skylake)
  Threads: 23 on 16 virtual cores
Environment:
  JULIA_EDITOR = code
```

### PLS with n = 1e6 observations

```julia
using Jchemo

## PLS2 with 1e6 observations
## (NB.: multi-threading is not used in plskern) 
n = 10^6  # nb. observations (samples)
p = 500   # nb. X-variables (features)
q = 10    # nb. Y-variables to predict
nlv = 25  # nb. PLS latent variables
X = rand(n, p)
Y = rand(n, q)
zX = Float32.(X)
zY = Float32.(Y)
```
```julia

## Float64
@benchmark fit!($mod, $X, $Y)

BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 7.532 s (1.07% GC) to evaluate,
 with a memory estimate of 4.09 GiB, over 2677 allocations.
```

```julia
# Float32 
@benchmark fit!($mod, $zX, $zY) 

BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  3.956 s …    4.148 s  ┊ GC (min … max): 0.82% … 3.95%
 Time  (median):     4.052 s               ┊ GC (median):    2.42%
 Time  (mean ± σ):   4.052 s ± 135.259 ms  ┊ GC (mean ± σ):  2.42% ± 2.21%

  █                                                        █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  3.96 s         Histogram: frequency by time         4.15 s <

 Memory estimate: 2.05 GiB, allocs estimate: 2677.
```

# <span style="color:green"> Examples of syntax </span> 

### **Examples of syntax** </span> 

#### **Fitting a model**

```julia
using Jchemo

n = 150 ; p = 200 ; q = 2 ; m = 50 
Xtrain = rand(n, p) ; Ytrain = rand(n, q) 
Xtest = rand(m, p) ; Ytest = rand(m, q) 

## Model fitting
nlv = 5 
fm = plskern(Xtrain, Ytrain; nlv) ;
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

#### **Tuning a model by grid-search** 

- #### With gridscore

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
    score = rmsep, fun = plskern, nlv) 

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

- #### With gridcv

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
    score = rmsep, fun = plskern, nlv) ;
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

# <span style="color:green"> Credit </span> 

### Author

**Matthieu Lesnoff**

- Cirad, [**UMR Selmet**](https://umr-selmet.cirad.fr/en), Montpellier, France

- [**ChemHouse**](https://www.chemproject.org/ChemHouse), Montpellier

**matthieu.lesnoff@cirad.fr**

###  How to cite

Lesnoff, M. 2021. Jchemo: Machine learning and chemometrics 
on high-dimensional data with Julia. https://github.com/mlesnoff/Jchemo. 
UMR SELMET, Univ Montpellier, CIRAD, INRA, Institut Agro, Montpellier, France

###  Acknowledgments

- G. Cornu (Cirad) https://ur-forets-societes.cirad.fr/en/l-unite/l-equipe
- M. Metz (Pellenc ST, Pertuis, France) 
- L. Plagne, F. Févotte (Triscale.innov) https://www.triscale-innov.com 
- R. Vezy (Cirad) https://www.youtube.com/channel/UCxArXLI-gxlTmWGGgec5D7w 



