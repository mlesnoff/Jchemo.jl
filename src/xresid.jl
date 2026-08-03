"""
    xresid(object, X, nlv::Int = nco(object.T))
    xresid!(object, X::Matrix{Q}, nlv::Int = nco(object.T)) where Q <: Float
Residual matrix from a bilinear model (e.g., PCA).
* `object` : The fitted model.
* `X` : X-data to be approximated from the model. Must be in the same scale as the X-data used to fit
    model `object`, i.e. before centering and eventual scaling.
Keyword arguments:
* `nlv` : Nb. components (PCs or LVs) to consider. By default, it is the maximum nb. of components
    defined in model `object`.

Compute the residual matrix:
* E = `X` - Xfit
where Xfit is the fitted X returned by function `xfit`. See `xfit` for examples. 
```
""" 
function xresid(object, X, nlv::Int = nco(object.T))
    xresid!(object, copy(ensure_mat(X)), nlv)
end

function xresid!(object, X::Matrix{Q}, nlv::Int) where Q <: Float
    nlv = min(nlv, object.par.nlv)
    X .-= xfit(object, X, nlv)
    X
end

