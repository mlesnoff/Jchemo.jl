"""
    xfit(object::Union{Pca, Pcr, Plsr}, X; nlv = nothing)
Matrix fitting from a PCA, PCR or PLS model
* `object` : The fitted model.
* `X` : X-data to be approximatred from the model.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, 
    it is the maximum nb. of components.

Compute an approximate of matrix `X` (X_fit) from a PCA, PCR 
or PLS fitted on `X`.

## Examples 
```julia 
n, p = 5, 3
X = rand(n, p)
y = rand(n)

nlv = 2 ;
fm = pcasvd(X; nlv = nlv) ;
#fm = plskern(X, y; nlv = nlv) ;
xfit(fm, X)
xfit(fm, X, nlv = 0)
xfit(fm, X, nlv = 1)

fm = pcasvd(X; nlv = min(n, p)) ;
xfit(fm, X)
xresid(fm, X)
```
""" 
function xfit(object::Union{Pca, Pcr, Plsr}, X; nlv = nothing)
    xfit!(object, copy(X); nlv = nlv)
end

function xfit!(object::Union{Pca, Pcr, Plsr}, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    isa(object, Jchemo.Pcr) ? object = object.fm_pca : nothing
    if nlv == 0
        m = size(X, 1)
        @inbounds for i = 1:m
            X[i, :] .= object.xmeans
        end
    else
        P = @view(object.P[:, 1:nlv])
        mul!(X, transform(object, X; nlv = nlv), P')
        center!(X, -object.xmeans)
    end
    X
end

"""
    xresid(object::Union{Pca, Pcr, Plsr}, X; nlv = nothing)
Residual matrix after fitting by a PCA, PCR or PLS model
* `object` : The fitted model.
* `X` : X-data for which the residuals have to be computed.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, 
    it is the maximum nb. of components.

Compute the residual matrix E = X - X_fit.

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
    xresid!(object, copy(X); nlv = nlv)
end

function xresid!(object::Union{Pca, Pcr, Plsr}, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    if(nlv == 0)
        center!(X, object.xmeans)
    else
        X .= X .- xfit(object, X; nlv = nlv)
    end
    X
end



