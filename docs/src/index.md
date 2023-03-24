```@meta
DocTestSetup  = quote
    using Jchemo
end
```

# Jchemo.jl

Documentation for [Jchemo.jl](https://github.com/mlesnoff/Jchemo.jl).

## Overview

**Jchemo.jl** is a [**package**](https://mlesnoff.github.io/Jchemo.jl/stable/domains/) 
for **data exploration and prediction** with focus on **high dimensional data**. 

The package was initially designed about **partial least squares regression and discrimination models** and variants, in particular locally weighted PLS models (**LWPLS**) (e.g. https://doi.org/10.1002/cem.3209).
Then, it has been expanded to many other methods for analyzing high dimensional data. 

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

## Examples of syntax for predictive models 

**Note:** More examples are given [**here**](https://github.com/mlesnoff/JchemoDemo).

### **Fitting a model**

```julia
using Jchemo

n = 150 ; p = 200 ; q = 2 ; m = 50 
Xtrain = rand(n, p) ; Ytrain = rand(n, q) ;
Xtest = rand(m, p) ; Ytest = rand(m, q) ;

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

```@autodocs
Modules = [Jchemo]
Order   = [:function, :type]
```

