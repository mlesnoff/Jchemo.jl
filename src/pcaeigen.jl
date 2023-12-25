"""
    pcaeigen(X, weights = ones(nro(X)); nlv, scal::Bool = false)
    pcaeigen!(X::Matrix, weights = ones(nro(X)); nlv, scal::Bool = false)
PCA by Eigen factorization.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
* `nlv` : Nb. principal components (PCs).
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Let us note D the (n, n) diagonal matrix of `weights`
and X the centered matrix in metric D. 
The function minimizes ||X - T * P'||^2  in metric D, by 
computing an Eigen factorization of X' * D * X. 

See `?pcasvd` for examples.
""" 
function pcaeigen(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    pcaeigen(X, weights; kwargs...)
end

function pcaeigen(X, weights::Weight; kwargs...)
    pcaeigen!(copy(ensure_mat(X)), weights; 
        kwargs...)
end

function pcaeigen!(X::Matrix, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs) 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = colmean(X, weights) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    sqrtw = sqrt.(weights.w)
    X .= Diagonal(sqrtw) * X
    res = eigen!(Symmetric(X' * X); sortby = x -> -abs(x)) 
    P = res.vectors[:, 1:nlv]
    eig = res.values[1:min(n, p)]
    eig[eig .< 0] .= 0
    sv = sqrt.(eig)
    T = Diagonal(1 ./ sqrtw) * X * P
    Pca(T, P, sv, xmeans, xscales, weights, nothing, kwargs, par) 
end

