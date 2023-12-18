"""
    xfit(fm::Union{Pca, Pcr, Plsr})
    xfit(fm::Union{Pca, Pcr, Plsr}, X; nlv = nothing)
    xfit!(fm::Union{Pca, Pcr, Plsr}, X::Matrix; nlv = nothing)
Matrix fitting from a PCA, PCR or PLS model
* `fm` : The fitted model.
* `X` : X-data to be approximatred from the model.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, 
    it is the maximum nb. of components.

Compute an approximate of matrix `X` (X_fit) from a PCA, PCR 
or PLS fitted on `X`.

`X` and X_fit are in the original fscale, i.e. before centering and eventual scaling.

## Examples 
```julia 
n, p = 5, 3
X = rand(n, p)
y = rand(n)

nlv = 2 ;
fm = pcasvd(X; nlv = nlv) ;
#fm = plskern(X, y; nlv = nlv) ;
xfit(fm)
xfit(fm, X)
xfit(fm, X, nlv = 0)
xfit(fm, X, nlv = 1)

fm = pcasvd(X; nlv = min(n, p)) ;
xfit(fm, X)
xresid(fm, X)
```
""" 
function xfit(fm)
    X = fm.T * fm.P'
    ## Coming back to the original scale
    fscale!(X, 1 ./ fm.xscales)    
    fcenter!(X, -fm.xmeans)
    X
end

function xfit(fm, X; nlv = nothing)
    xfit!(fm, copy(ensure_mat(X)); 
        nlv)
end

function xfit!(fm, X::Matrix; 
        nlv = nothing)
    a = nco(fm.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    if nlv == 0
        m = nro(X)
        @inbounds for i = 1:m
            X[i, :] .= fm.xmeans
        end
    else
        P = vcol(fm.P, 1:nlv)
        mul!(X, transf(fm, X; nlv), P')
        ## Coming back to the original scale
        fscale!(X, 1 ./ fm.xscales)    
        fcenter!(X, -fm.xmeans)
    end
    X
end
