```@meta
DocTestSetup  = quote
    using Jchemo
end
```

# Jchemo.jl

Documentation for [Jchemo.jl](https://github.com/mlesnoff/Jchemo.jl).

## Overview

**Jchemo** provides [**functions**](https://github.com/mlesnoff/Jchemo.jl/blob/master/docs/src/domains.md) 
for data exploration and predictions in chemometrics or other domains, with focus on high dimensional data. 

The package was initially designed about **k-nearest neighbors locally weighted partial least squares regression 
and discrimination models** (kNN-LWPLSR and kNN-LWPLSDA; e.g. https://doi.org/10.1002/cem.3209).
It has now been expanded to many other methods for analyzing high dimensional data. 

Generic functions such as **transform**, **predict**, **coef** and **summary** are available. 
**Tuning the predicive models** is facilitated by functions **gridscore** (validation dataset) and 
**gridcv** (cross-validation). Faster versions are also available for models based on latent variables (LVs) 
(**gridscorelv** and **gridcvlv**) and ridge regularization (**gridscorelb** and **gridcvlb**).

Some of the functions of **Jchemo** (in particular the function using kNN selections) use multi-threading 
to speed the computations. To take advantage of thos, the user has to specify his relevant number 
of threads (e.g. from the setting menu of the VsCode Julia extension and the file settings.json).

**Jchemo** uses **Makie** for plotting. To display the plots, the user has to preliminary install and load one 
of the Makie's backends (e.g. **CairoMakie**). 

Most of the functions have a **help page** (each given an example), e.g.:

```julia
?savgol
```

```@autodocs
Modules = [Jchemo]
Order   = [:function, :type]
```

