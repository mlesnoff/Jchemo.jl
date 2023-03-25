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


```@autodocs
Modules = [Jchemo]
Order   = [:function, :type]
```

