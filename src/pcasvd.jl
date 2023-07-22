struct Pca
    T::Array{Float64} 
    P::Array{Float64}
    sv::Vector{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    weights::Vector{Float64}
    ## For consistency with PCA Nipals
    niter::Union{Vector{Int64}, Nothing}
end

"""
    pcasvd(X, weights = ones(nro(X)); nlv, scal::Bool = false)
    pcasvd!(X::Matrix, weights = ones(nro(X)); nlv, scal::Bool = false)
PCA by SVD factorization.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. principal components (PCs).
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

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
using JchemoData, JLD2, CairoMakie, StatsBase
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
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

Jchemo.transform(fm, Xtest)

res = Base.summary(fm, Xtrain) ;
pnames(res)
res.explvarx
res.contr_var
res.coord_var
res.cor_circle
```
""" 
function pcasvd(X, weights = ones(nro(X)); nlv, 
        scal::Bool = false)
    pcasvd!(copy(ensure_mat(X)), weights; nlv = nlv, 
        scal = scal)
end

function pcasvd!(X::Matrix, weights = ones(nro(X)); nlv, 
        scal::Bool = false)
    n, p = size(X)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    xmeans = colmean(X, weights)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X, weights)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    ## by default in LinearAlgebra.svd
    ## "full = false" ==> [1:min(n, p)]
    sqrtw = sqrt.(weights)
    res = LinearAlgebra.svd!(sqrtw .* X)
    P = res.V[:, 1:nlv]
    sv = res.S   
    sv[sv .< 0] .= 0
    T = (1 ./ sqrtw) .* vcol(res.U, 1:nlv) * (Diagonal(sv[1:nlv]))
    Pca(T, P, sv, xmeans, xscales, weights, nothing)
end

""" 
    transform(object::Pca, X; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transform(object::Union{Pca, Fda}, X; nlv = nothing)
    X = ensure_mat(X)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    cscale(X, object.xmeans, object.xscales) * vcol(object.P, 1:nlv)
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
    X = cscale(X, object.xmeans, object.xscales)
    sstot = sum(colnorm(X, object.weights).^2)   # = tr(X' * D * X)
    TT = D * object.T.^2
    tt = colsum(TT) 
    # = diag(T' * D * T) 
    # = colnorm(object.T, object.weights).^2 
    # = object.sv[1:nlv].^2
    pvar = tt / sstot
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    nam = string.("lv", 1:nlv)
    contr_ind = DataFrame(scale(TT, tt), nam)
    cor_circle = DataFrame(corm(X, object.T, object.weights), nam)
    C = X' * D * scale(object.T, sqrt.(tt))
    coord_var = DataFrame(C, nam)
    CC = C .* C
    cc = sum(CC, dims = 1)
    contr_var = DataFrame(scale(CC, cc), nam)
    (explvarx = explvarx, contr_ind, contr_var, coord_var, cor_circle)
end




