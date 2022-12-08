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

The package was initially about **k-nearest neighbors locally weighted partial least squares regression 
and discrimination models** (kNN-LWPLSR and kNN-LWPLSDA; e.g. Lesnoff et al 2021 https://doi.org/10.1002/cem.3209).
It has now been expanded to various other methods. 

Generic functions such as **transform**, **predict**, **coef** and **summary** are available. 
**Tuning the predicive models** is facilitated by functions **gridscore** (validation dataset) and 
**gridcv** (cross-validation). Faster versions are also available for models based on latent variables (LVs) 
(**gridscorelv** and **gridcvlv**) and ridge regularization (**gridscorelb** and **gridcvlb**).

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

