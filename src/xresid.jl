"""
    xresid(object, X; nlv::Union{Nothing, Int} = nothing)
    xresid!(object, X::Matrix; nlv::Union{Nothing, Int} = nothing)
Residual matrix from a bilinear model (e.g., PCA).
* `object` : The fitted model.
* `X` : New X-data to be approximated from the model.Must be in the same scale as the X-data used to fit
    the model `object`, i.e. before centering and eventual scaling.
Keyword arguments:
* `nlv` : Nb. components (PCs or LVs) to consider. If `nothing`, it is the maximum nb. of components.

Compute the residual matrix:
* E = `X` - X_fit
where X_fit is the fitted X returned by function `xfit`. See `xfit` for examples. 
```
""" 
function xresid(object, X; nlv::Union{Nothing, Int} = nothing)
    xresid!(object, copy(ensure_mat(X)); nlv)
end

function xresid!(object, X::Matrix; nlv::Union{Nothing, Int} = nothing)
    a = object.par.nlv
    nlv = isnothing(nlv) ? a : min(nlv, a)
    X .-= xfit(object, X; nlv)
    X
end

