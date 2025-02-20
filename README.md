# Jchemo.jl

### Chemometrics and machine learning on high-dimensional data with Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/dev)
[![Build Status](https://github.com/mlesnoff/Jchemo.jl/workflows/CI/badge.svg)](https://github.com/mlesnoff/Jchemo.jl/actions)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

# <span style="color:green"> **About** </span> 

**Jchemo** was initially dedicated to **partial least squares regression (PLSR) and discrimination (PLSDA)** 
and their extensions, in particular locally weighted PLS models (**KNN-LWPLS-R** & **-DA**; e.g. https://doi.org/10.1002/cem.3209). In a second phase, the package has expand to many other models for **dimension reduction** and **regression and discrimination** ([see the list of functions here](https://mlesnoff.github.io/Jchemo.jl/dev/domains/)). 

Why the name **Jchemo**?: Since it is orientated to **chemometrics** , in brief the use of biometrics for chemistry. But most of the provided methods are generic to other types of data than chemistry. 

**Jchemo** is organized between 
- **transform operators** (that have a function `transf`),
- **predictors** (that have a function `predict`), 
- **utility functions**. 

Some models, such as PLSR models, are both a transform operator and a predictor.

Ad'hoc **pipelines** of operations can also be built. 
In **Jchemo**, a pipeline is a **chain of *K* modeling steps** containing
- either ***K* transform steps**,
- or ***K* - 1 transform steps** and **a final prediction step**. 

The pipelines are built with function `pip`, see [**here**](https://github.com/mlesnoff/Jchemo.jl/tree/master?tab=readme-ov-file#fitting-a-pipeline).

# <span style="color:green"> **Installation** </span> 

To install **Jchemo** 

* From the official Julia repo, run in the Pkg REPL:
```julia
pkg> add Jchemo
```

or for a **specific version**, for instance: 
```julia
pkg> add Jchemo@0.1.18
```

* For the **current developing version** (potentially not stable):
```julia
pkg> add https://github.com/mlesnoff/Jchemo.jl.git
```
### Warning

Before to update the package, it is recommended to have a look on [**What changed**](https://mlesnoff.github.io/Jchemo.jl/dev/news/) to avoid eventual problems due to breaking changes. 

# <span style="color:green"> **Tips** </span> 

### Syntax

Most the **Jchemo** functions have **keyword arguments** (`kwargs`). The keyword arguments required/allowed in a function can be found in the **Index of function section** of the documentation:
- [Stable](https://mlesnoff.github.io/Jchemo.jl/stable/api/) 
- [Developping](https://mlesnoff.github.io/Jchemo.jl/dev/api/) 

or in the REPL by displaying the function's help page, for instance for function `plskern`:

```julia
julia> ?plskern
```

The `kwargs` arguments and their default values are defined in containers depending of the functions, available 
[**here**](https://github.com/mlesnoff/Jchemo.jl/blob/master/src/_structures_param.jl). For a given function, they can be displayed in the REPL with macro `@pars`, e.g.:

```julia
julia> @pars plskern

Jchemo.ParPlsr
  nlv: Int64 1
  scal: Bool false
```

**Examples of syntax** are given at the end of this README, and in the help pages of the functions.

### Datasets

The **datasets** used in the help page examples are stored in the package [**JchemoData.jl**](https://github.com/mlesnoff/JchemoData.jl), a repository of datasets on chemometrics and other domains. **Examples of scripts** demonstrating **Jchemo** are also available in the pedagogical project [**JchemoDemo**](https://github.com/mlesnoff/JchemoDemo). 

### Tuning predictive models

Two generic **grid-search** functions are available to tune the predictors: 
- [`gridscore`](https://mlesnoff.github.io/Jchemo.jl/stable/api/#Jchemo.gridscore-NTuple{5,%20Any}) (*test-set* validation)
- [`gridcv`](https://mlesnoff.github.io/Jchemo.jl/stable/api/#Jchemo.gridcv-Tuple{Any,%20Any,%20Any}) (cross-validation). 

Highly accelerated versions of these tuning tools have been implemented for models based on latent variables (LVs) and ridge regularization.

### Multi-threading

Some **Jchemo** functions (e.g. those using kNN selections) use **multi-threading** 
to speed the computations. Taking advantage of this requires to specify a relevant number 
of threads (for instance from the *Settings* menu of the VsCode Julia extension and the file *settings.json*).

### Plotting

**Jchemo** uses **Makie** for plotting. Displaying the plots requires to install and load one of the Makie's backends (e.g. **CairoMakie**). 

# <span style="color:green">  **Benchmark**  </span>

```julia
using Jchemo, BenchmarkTools
```

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

### Multi-variate PLSR with n = 1e6 observations

```julia
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
## (NB.: multi-threading is not used in plskern) 
model = plskern(; nlv)
@benchmark fit!($model, $X, $Y)

BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 7.532 s (1.07% GC) to evaluate,
 with a memory estimate of 4.09 GiB, over 2677 allocations.
```

```julia
# Float32 
@benchmark fit!($model, $zX, $zY) 

BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  3.956 s …    4.148 s  ┊ GC (min … max): 0.82% … 3.95%
 Time  (median):     4.052 s               ┊ GC (median):    2.42%
 Time  (mean ± σ):   4.052 s ± 135.259 ms  ┊ GC (mean ± σ):  2.42% ± 2.21%

  █                                                        █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  3.96 s         Histogram: frequency by time         4.15 s <

 Memory estimate: 2.05 GiB, allocs estimate: 2677.
```

# <span style="color:green"> **Examples of syntax** </span> 

### Some fictive data

```julia
n = 150 ; p = 200 
q = 2 ; m = 50 
Xtrain = rand(n, p)
Ytrain = rand(n, q) 
Xtest = rand(m, p)
Ytest = rand(m, q) 
```

### **Fitting a transform model**

#### **a) Example of a signal preprocessing**

Let us consider a signal preprocessing with the Savitsky-Golay filter, using function `savgol`. The keyword arguments of `savgol` are `npoint`, `deriv` and `degree`. See for instance in the REPL:

```julia
julia> ?savgol
```

The syntax to fit the model is as follows:

```julia
## Below, the order of the kwargs is not 
## important but the argument names have 
## to be correct.
## Keywords arguments are specified
## after character ";"

## Model definition
## (below, the name 'model' can be replaced by any other name)
npoint = 11 ; deriv = 2 ; degree = 3
model = savgol(; npoint, deriv, degree)

## Fitting
fit!(model, Xtrain)
```

which is the strictly equivalent to:

```julia
## Below, ";" is not required since the kwargs values are
## specified within the function

model = savgol(npoint = 11, deriv = 2, degree = 3)
fit!(model, Xtrain)
```

Contents of object `model` can be displayed by:

``` julia
julia> @names model

(:algo, :fitm, :kwargs)
```

Sub-objects `algo`, `fitm` and `kwargs` contain the used algorithm (a function), the model fitted on the data, and the kwargs arguments, respectively.

Once the model is fitted, the transformed data are given by:

```julia
Xptrain = transf(model, Xtrain)   # preprocessed data
Xptest = transf(model, Xtest)
```

Several preprocessing can be applied sequentially to the data by building a **pipeline** (see section *Fitting a pipeline* thereafter for examples).

#### **b) Example of a PCA**

Let us consider a principal component analysis (PCA), using function `pcasvd`. 

The syntax to fit the model using a SVD decomposition is as follows:
```julia
nlv = 15  # nb. principal components
model = pcasvd(; nlv)
fit!(model, Xtrain, ytrain)
```

The PCA score matrices (i.e. the projections of the observations on the PCA directions) can be computed by:
```julia
Ttrain = transf(model, Xtrain)
Ttest = transf(model, Xtest)
```

Object `Ttrain` above can also be obtained directly by:

```julia
Ttrain = model.fitm.T
```

Some model summary (% of explained variance, etc.) can be displayed by:

```julia
summary(model, Xtrain)
```

For a preliminary scaling of the data before the PCA decomposition, the syntax is:

```julia
nlv = 15 ; scal = true
model = pcasvd(; nlv, scal)
fit!(model, Xtrain, ytrain)
```

### **Fitting a prediction model**

#### **a) Example of a KPLSR**

Let us consider a Gaussian kernel partial least squares regression (KPLSR), using function `kplsr`. 

The syntax to fit the model is as follows:
```julia
nlv = 15  # nb. latent variables
kern = :krbf ; gamma = .001 
model = kplsr(; nlv, kern, gamma)
fit!(model, Xtrain, ytrain)
```

As for PCA, the score matrices can be computed by:
```julia
Ttrain = transf(model, Xtrain)   # = model.fitm.T
Ttest = transf(model, Xtest)
```

and model summary by:

```julia
summary(model, Xtrain)
```

Y-Predictions are given by:
```julia
pred = predict(model, Xtest).pred
```

**Examples of tuning** of predictive models (test-set validation and K-fold cross-validation) are given in the help pages of functions `gridscore` and `gridcv`: 

```julia
?gridscore
?gridcv
```
### **Fitting a pipeline**

#### **a) Example of chained preprocessing**

Let us consider a data preprocessing by standard-normal-variation transformation (SNV) followed by a Savitsky-Golay filter and a polynomial de-trending transformation. 

The pipeline is fitted as follows:

```julia
## Models' definition
model1 = snv()
model2 = savgol(npoint = 5, deriv = 1, degree = 2)
model3 = detrend_pol()  
## Pipeline building
model = pip(model1, model2, model3)
## Fitting
fit!(model, Xtrain)
```

The transformed data are given by:

```julia
Xptrain = transf(model, Xtrain)
Xptest = transf(model, Xtest)
```
#### **b) Example of PCA-SVMR**

Let us consider a support vector machine regression model implemented on preliminary computed PCA scores (PCA-SVMR). 

The pipeline is fitted as follows:

```julia
nlv = 15
kern = :krbf ; gamma = .001 ; cost = 1000
model1 = pcasvd(; nlv)
model2 = svmr(; kern, gamma, cost)
model = pip(model1, model2)
fit!(model, Xtrain)
```

The Y-predictions are given by:
```julia
pred = predict(model, Xtest).pred
```

Any step(s) of data preprocessing can obviously be implemented before the modeling, either outside of the given predictive pipeline or being involded directlty in the pipeline, such as for instance:

```julia
degree = 2    # de-trending with polynom degree 2
nlv = 15
kern = :krbf ; gamma = .001 ; cost = 1000
model1 = detrend_pol(; degree)
model2 = pcasvd(; nlv)
model3 = svmr(; kern, gamma, cost)
model = pip(model1, model2, model3)
```

#### **c) Example of LWR Naes et al. 1990**

The LWR algorithm of Naes et al (1990) consists in implementing a preliminary global PCA on the data and then a kNN locally weighted multiple linear regression (kNN-LWMLR) on the global PCA scores.

The pipeline is defined by:

```julia
nlv = 25
metric = :eucl ; h = 2 ; k = 200
model1 = pcasvd(; nlv)
model2 = lwmlr(; metric, h, k)
model = pip(model1, model2)
```

*Naes et al., 1990. Analytical Chemistry 664–673.*

#### **d) Example of Shen et al. 2019**

The pipeline of Shen et al. (2019) consists in implementing a preliminary global PLSR on the data and then a kNN-PLSR on the global PLSR scores.

The pipeline is defined by:

```julia
nlv = 25
metric = :mah ; h = Inf ; k = 200
model1 = plskern(; nlv)
model2 = lwplsr(; metric, h, k)
model = pip(model1, model2)
```

*Shen et al., 2019. Journal of Chemometrics, 33(5) e3117.*

# <span style="color:green"> **Credit** </span> 

### **Author**

Matthieu Lesnoff     
contact: **matthieu.lesnoff@cirad.fr**

- Cirad, [**UMR Selmet**](https://umr-selmet.cirad.fr/en), Montpellier, France

- [**ChemHouse**](https://www.chemproject.org/ChemHouse), Montpellier

### **How to cite**

Lesnoff, M. 2021. Jchemo: Chemometrics and machine learning on high-dimensional data with Julia. https://github.com/mlesnoff/Jchemo. 
UMR SELMET, Univ Montpellier, CIRAD, INRA, Institut Agro, Montpellier, France

###  **Acknowledgments**

- G. Cornu (Cirad) https://ur-forets-societes.cirad.fr/en/l-unite/l-equipe
- M. Metz (Pellenc ST, Pertuis, France) 
- L. Plagne, F. Févotte (Triscale.innov) https://www.triscale-innov.com 
- R. Vezy (Cirad) https://www.youtube.com/channel/UCxArXLI-gxlTmWGGgec5D7w 



