struct Pca
    T::Array{Float64} 
    P::Array{Float64}
    sv::Vector{Float64}
    xmeans::Vector{Float64}
    weights::Vector{Float64}
    ## For consistency with PCA Nipals
    niter::Union{Int64, Nothing}
    conv::Union{Bool, Nothing}
end

"""
    pcasvd(X, weights = ones(size(X, 1)); nlv)
    pcasvd!(X::Matrix, weights = ones(size(X, 1)); nlv)
PCA by SVD factorization.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations.
* `nlv` : Nb. principal components (PCs).

`weights` is internally normalized to sum to 1.

Let us note D the (n, n) diagonal matrix of `weights`
and X the centered matrix in metric D.
The function minimizes ||X - T * P'||^2  in metric D, by 
computing a SVD factorization of sqrt(D) * X:

* sqrt(D) * X ~ U * S * V'

Outputs are:

* T = D^(-1/2) * U * S
* P = V
* The diagonal of S   

## Examples
```julia
using JLD2, CairoMakie, StatsBase
mypath = dirname(dirname(pathof(Jchemo)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
pnames(dat)
summ(dat.X)

X = dat.X[:, 1:4]
n = nro(X)

ntrain = 120
s = sample(1:n, ntrain; replace = false) 
Xtrain = X[s, :]
Xtest = rmrow(X, s)

nlv = 3
fm = pcasvd(Xtrain; nlv = nlv) ;
#fm = pcaeigen(Xtrain; nlv = nlv) ;
#fm = pcaeigenk(Xtrain; nlv = nlv) ;
pnames(fm)
fm.T
fm.T' * fm.T
fm.P' * fm.P

transform(fm, Xtest)

res = Base.summary(fm, Xtrain) ;
pnames(res)
res.explvar
res.contr_var
res.coord_var
res.cor_circle
```
""" 
function pcasvd(X, weights = ones(size(X, 1)); nlv)
    pcasvd!(copy(ensure_mat(X)), weights; nlv = nlv)
end

function pcasvd!(X::Matrix, weights = ones(size(X, 1)); nlv)
    n, p = size(X)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    center!(X, xmeans)
    ## by default in LinearAlgebra.svd
    ## "full = false" ==> [1:min(n, p)]
    sqrtw = sqrt.(weights)
    res = LinearAlgebra.svd!(Diagonal(sqrtw) * X)
    P = res.V[:, 1:nlv]
    sv = res.S   
    sv[sv .< 0] .= 0
    T = Diagonal(1 ./ sqrtw) * vcol(res.U, 1:nlv) * (Diagonal(sv[1:nlv]))
    Pca(T, P, sv, xmeans, weights, nothing, nothing)
end

"""
    summary(object::Pca, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Pca, X::Union{Matrix, DataFrame})
    X = ensure_mat(X)
    nlv = size(object.T, 2)
    D = Diagonal(object.weights)
    X = center(X, object.xmeans)
    sstot = sum(colnorm2(X, object.weights))   # = tr(X' * D * X)
    TT = D * object.T.^2
    tt = colsum(TT) 
    # = diag(T' * D * T) 
    # = colnorm2(object.T, object.weights) 
    # = object.sv[1:nlv].^2
    pvar = tt / sstot
    cumpvar = cumsum(pvar)
    explvar = DataFrame(pc = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    nam = string.("pc", 1:nlv)
    contr_ind = DataFrame(scale(TT, tt), nam)
    cor_circle = DataFrame(corm(X, object.T, object.weights), nam)
    C = X' * D * scale(object.T, sqrt.(tt))
    coord_var = DataFrame(C, nam)
    CC = C .* C
    cc = sum(CC, dims = 1)
    contr_var = DataFrame(scale(CC, cc), nam)
    (explvar = explvar, contr_ind, contr_var, coord_var, cor_circle)
end

""" 
    transform(object::Pca, X; nlv = nothing)
Compute components (scores matrix "T") from a fitted model and X-data.
* `object` : The maximal fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. components to compute. If nothing, it is the maximum nb. PCs.
""" 
function transform(object::Union{Pca, Fda}, X; nlv = nothing)
    X = ensure_mat(X)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    center(X, object.xmeans) * vcol(object.P, 1:nlv)
end



