"""
    xresid(object::Union{Pca, Pcr, Plsr}, X; nlv = nothing)
    xresid!(object::Union{Pca, Pcr, Plsr}, X::Matrix; nlv = nothing)
Residual matrix after fitting by a PCA, PCR or PLS model
* `object` : The fitted model.
* `X` : X-data for which the residuals have to be computed.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, 
    it is the maximum nb. of components.

Compute the residual matrix E = X - X_fit.

`X` and X_fit are in the original scale, i.e. before centering and eventual scaling.

## Examples 
```julia 
n, p = 5, 3
X = rand(n, p)
y = rand(n)

nlv = 2 ;
fm = pcasvd(X; nlv = nlv) ;
#fm = plskern(X, y; nlv = nlv) ;
xresid(fm, X)
xresid(fm, X, nlv = 0)
xresid(fm, X, nlv = 1)

fm = pcasvd(X; nlv = min(n, p)) ;
xfit(fm, X)
xresid(fm, X)
```
""" 
function xresid(object::Union{Pca, Pcr, Plsr}, X; nlv = nothing)
    xresid!(object, copy(ensure_mat(X)); nlv = nlv)
end

function xresid!(object::Union{Pca, Pcr, Plsr}, X::Matrix; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    X .-= xfit(object, X; nlv = nlv)
    X
end

