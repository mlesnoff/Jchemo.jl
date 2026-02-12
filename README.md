# Jchemo.jl

### Chemometrics and machine learning for high-dimensional data with Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlesnoff.github.io/Jchemo.jl/dev)
[![Build Status](https://github.com/mlesnoff/Jchemo.jl/workflows/CI/badge.svg)](https://github.com/mlesnoff/Jchemo.jl/actions)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

# <span style="color:green"> **About** </span> 

At the start of the project, **Jchemo** was built around **partial least squares regression (PLSR) and discrimination (PLSDA) 
methods** and their non-linear extensions, in particular locally weighted PLS models (**kNN-LWPLS-R** & **-DA**; e.g., https://doi.org/10.1002/cem.3209). The package has then been expanded with [many other methods](https://mlesnoff.github.io/Jchemo.jl/dev/domains/) of dimension reduction, regression, discrimination, and signal (e.g., spectra) preprocessing. 

Why the name **Jchemo**? Since it is oriented towards **chemometrics**, in brief the use of biometrics for chemistry data. But most of the provided methods are generic and can be applied to other types of data. 

Related projects:  [JchemoData.jl](https://github.com/mlesnoff/JchemoData.jl) (a "container" package gathering selected data sets used in the examples) and [JchemoDemo](https://github.com/mlesnoff/JchemoDemo) (a pedagogical environment).

*Warning*

Before to [update](https://github.com/mlesnoff/Jchemo.jl?tab=readme-ov-file#-installation-) the package, it is recommended to have a look on [What changed](https://mlesnoff.github.io/Jchemo.jl/dev/news/) for eventual breaking changes. 

# <span style="color:green"> **Sample workflow** </span> 

Let us assume training data `(X, Y)`, and new data `Xnew` for which we want predictions from a PLSR model with 15 latent variables (LVs). The workflow is has follows 
1) An object, e.g., `model` (any other name can be chosen), is built from the given learning model and its eventual parameters.
    This object contains three sub-objects 
    * `algo` (the learning algorithm) 
    * `fitm` (the fitted model, empty at this stage) 
    * and `kwargs` (the specified keyword arguments).
2) Function `fit!` fits the model to the training data, which fills sub-object `fitm` above.
3) Function `predict` computes the predictions.   

```julia
model = plskern(nlv = 15, scal = true)
fit!(model, X, Y)
pred = predict(model Xnew).pred  # '.pred' is specified here since functio 'predict' can return several objects for some models 
```

We can check the contents of object `model`

``` julia
@names model

(:algo, :fitm, :kwargs)
```

An alternative syntax to specify the keyword arguments is 

```julia
nlv = 15 ; scal = true
model = plskern(; nlv, scal)
```

The default values of the keyword arguments of the model function can be displayed using macro `@pars`

```julia
@pars plskern

Jchemo.ParPlsr
  nlv: Int64 1
  scal: Bool false
```

After model fitting, the matrices of the PLS scores can be obtained from function `transf`

```julia
T = transf(model, X)   # can also be obtained directly by: model.fitm.T
Tnew = transf(model, Xnew)
```

Other examples of workflows are given at the [end](https://github.com/mlesnoff/Jchemo.jl?tab=readme-ov-file#-examples-of-syntax-) of this README.

# <span style="color:green"> **Package structure** </span> 

**Jchemo** is organized between 
- **transform operators** (that have a function `transf`)
- **predictors** (that have a function `predict`) 
- **utility functions**. 

Some models, such as PLSR models, are both a transform operator and a predictor.

**Ad'hoc pipelines** of operations can also be built, using function [`pip`](https://github.com/mlesnoff/Jchemo.jl/tree/master?tab=readme-ov-file#pipelines). In Jchemo, a pipeline is a **chain of *K* modeling steps** containing
- either ***K* transform steps**
- or ***K* - 1 transform steps** and **a final prediction step**. 

*Keyword arguments*

The keyword arguments required or allowed in a given function can be found in the [here](https://mlesnoff.github.io/Jchemo.jl/stable/api/) or in the REPL by displaying the function's help page. For instance for function [`plskern`](https://mlesnoff.github.io/Jchemo.jl/stable/api/#Jchemo.plskern-Tuple{})

```julia
julia> ?plskern
```

Default values can be displayed in the REPL with macro `@pars`

```julia
julia> @pars plskern

Jchemo.ParPlsr
  nlv: Int64 1
  scal: Bool false
```

*Multi-threading*

Some functions (e.g., those using kNN selections) use **multi-threading** to speed the computations. Taking advantage of this requires to specify a relevant number of threads (for instance from the *Settings* menu of the VsCode Julia extension and the file *settings.json*).

*Plotting*

**Jchemo** uses **Makie** for plotting. Displaying the plots requires to install and load one of the Makie's backends (**CairoMakie** or **GLMakie**). 

*Datasets*

The **datasets** used as examples in the function help pages are stored in package [JchemoData.jl](https://github.com/mlesnoff/JchemoData.jl), a repository of datasets on chemometrics and other domains. **Examples of scripts** demonstrating **Jchemo** are also available in the pedagogical project [JchemoDemo](https://github.com/mlesnoff/JchemoDemo). 

# <span style="color:green"> **Tuning predictive models** </span> 

Two **grid-search** functions are available to tune the predictors 
- [`gridscore`](https://mlesnoff.github.io/Jchemo.jl/stable/api/#Jchemo.gridscore-NTuple{5,%20Any}): tuning using a partition calibration/validation
- [`gridcv`](https://mlesnoff.github.io/Jchemo.jl/stable/api/#Jchemo.gridcv-Tuple{Any,%20Any,%20Any}): tuning using a cross-validation (e.g., K-fold) process. 

The syntax is generic for all the predictors (see the help pages of the two functions above for sample workflows). The two functions have been specifically accelerated (using computation tricks) for models based either on latent variables or ridge regularization.

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
model = plskern(; nlv)
@benchmark fit!($model, $X, $Y)

BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 7.532 s (1.07% GC) to evaluate,
 with a memory estimate of 4.09 GiB, over 2677 allocations.
```

```julia
## Float32 
@benchmark fit!($model, $zX, $zY) 

BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  3.956 s …    4.148 s  ┊ GC (min … max): 0.82% … 3.95%
 Time  (median):     4.052 s               ┊ GC (median):    2.42%
 Time  (mean ± σ):   4.052 s ± 135.259 ms  ┊ GC (mean ± σ):  2.42% ± 2.21%

  █                                                        █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  3.96 s         Histogram: frequency by time         4.15 s <

 Memory estimate: 2.05 GiB, allocs estimate: 2677.

## (NB.: multi-threading is not used in plskern) 
```

# <span style="color:green"> **Installation** </span> 

To install **Jchemo** 

* From the official Julia repo, run in the Pkg REPL

```julia
pkg> add Jchemo
```

or for a **specific version**, for instance 

```julia
pkg> add Jchemo@0.9.1
```

* For the **current developing version** (potentially not stable)

```julia
pkg> add https://github.com/mlesnoff/Jchemo.jl.git
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

### **Transform operations**

#### **a) Example of signal preprocessing**

Consider a signal preprocessing with the Savitsky-Golay filter, using function `savgol`

```julia
## Below, the order of the kwargs is not important but the argument 
## names have to be correct.

## Model definition
## (below, the name 'model' can be replaced by any other name)
npoint = 11 ; deriv = 2 ; degree = 3
model = savgol(; npoint, deriv, degree)

## Fitting
fit!(model, Xtrain)

## Transformed (= preprocessed) data
Xptrain = transf(model, Xtrain)  
Xptest = transf(model, Xtest)
```

Several preprocessing can be applied sequentially to the data by building a [pipeline](https://github.com/mlesnoff/Jchemo.jl/tree/master?tab=readme-ov-file#fitting-a-pipeline).

#### **b) Example of PCA**

Consider a SVD principal component analysis 

```julia
nlv = 15  # nb. principal components
model = pcasvd(; nlv)
fit!(model, Xtrain, ytrain)

## Score matrices
Ttrain = transf(model, Xtrain) # same as:  model.fitm.T
Ttest = transf(model, Xtest)

## Model summary (% of explained variance, etc.)
summary(model, Xtrain)
```

For a preliminary scaling of the data before the PCA

```julia
nlv = 15 ; scal = true
model = pcasvd(; nlv, scal)
fit!(model, Xtrain, ytrain)
```

### **Prediction models**

#### **a) Example of KPLSR**

Consider a Gaussian kernel partial least squares regression (KPLSR), using function `kplsr` 

```julia
nlv = 15  # nb. latent variables
kern = :krbf ; gamma = .001 
model = kplsr(; nlv, kern, gamma)
fit!(model, Xtrain, ytrain)

## PLS score matrices can be computed by:
Ttrain = transf(model, Xtrain)   # = model.fitm.T
Ttest = transf(model, Xtest)

## Model summary
summary(model, Xtrain)

## Y-Predictions
pred = predict(model, Xtest).pred
```

### **Pipelines**

#### **a) Example of chained preprocessing**

Consider a data preprocessing by standard-normal-variation transformation (SNV) followed by a Savitsky-Golay filter and a polynomial de-trending transformation

```julia
## Model definitions
model1 = snv()
model2 = savgol(npoint = 5, deriv = 1, degree = 2)
model3 = detrend_pol()  

## Pipeline building and fitting
model = pip(model1, model2, model3)
fit!(model, Xtrain)

## Transformed data
Xptrain = transf(model, Xtrain)
Xptest = transf(model, Xtest)
```
#### **b) Example of PCA-SVMR**

Consider a support vector machine regression model implemented on preliminary computed PLS scores (PLS-SVMR) 

```julia
nlv = 20
kern = :krbf
gamma = 1e4 ; cost = 1000 ; epsilon = .7
model1 = plskern(; nlv)
model2 = svmr(; kern, gamma, cost, epsilon)
model = pip(model1, model2)
fit!(model, Xtrain, ytrain)

## Y-predictions
pred = predict(model, Xtest).pred
```

Step(s) of data preprocessing can obviously be implemented before the model(s)

```julia
nlv = 20
kern = :krbf
gamma = 1e4 ; cost = 1000 ; epsilon = .7
model1 = detrend_pol(degree = 2)   # polynomial de-trending with polynom degree = 2
model2 = plskern(; nlv)
model3 = svmr(; kern, gamma, cost, epsilon)
model = pip(model1, model2, model3)
```

#### **c) Example of LWR (Naes et al. 1990)**

The LWR algorithm of Naes et al (1990) consists in implementing a preliminary global PCA on the data and then a kNN locally weighted multiple linear regression (kNN-LWMLR) on the global PCA scores

```julia
nlv = 25
metric = :eucl ; h = 2 ; k = 200
model1 = pcasvd(; nlv)
model2 = lwmlr(; metric, h, k)
model = pip(model1, model2)
```

*Naes et al., 1990. Analytical Chemistry 664–673.*

#### **d) Example of Shen et al. 2019**

The pipeline of Shen et al. (2019) consists in implementing a preliminary global PLSR on the data and then a kNN-PLSR on the global PLSR scores

```julia
nlv = 25
metric = :mah ; h = Inf ; k = 200
model1 = plskern(; nlv)
model2 = lwplsr(; metric, h, k, nlv = 10)
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

Lesnoff, M. 2021. Jchemo: Chemometrics and machine learning for high-dimensional data with Julia. https://github.com/mlesnoff/Jchemo.jl. 
UMR SELMET, Univ Montpellier, CIRAD, INRA, Institut Agro, Montpellier, France

###  **Acknowledgments**

- G. Cornu (Cirad) https://ur-forets-societes.cirad.fr/en/l-unite/l-equipe
- M. Metz (Pellenc ST, Pertuis, France) 
- L. Plagne, F. Févotte (Triscale.innov) https://www.triscale-innov.com 
- R. Vezy (Cirad) https://www.youtube.com/channel/UCxArXLI-gxlTmWGGgec5D7w 



