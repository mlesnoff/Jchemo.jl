```@meta
DocTestSetup  = quote
    using Jchemo
end
```

# Jchemo.jl

Documentation for [Jchemo.jl](https://github.com/mlesnoff/Jchemo.jl).

## Overview

**Jchemo** provides elementary functions and pipelines for predictions in chemometrics or other domains,
with focus on high dimensional data. 

Generic functions such as **transform**, **predict**, **coef** and **summary** are available. Tuning the models is facilitated by functions **gridscore** (validation dataset) and **gridcv** (cross-validation), in addition to faster versions for models based on latent variables (LVs) (**gridscorelv** and **gridcvlv**) and ridge regularization (**gridscorelb** and **gridcvlb**).

See the **News** section for eventual changes.

```@autodocs
Modules = [Jchemo]
Order   = [:function, :type]
```

