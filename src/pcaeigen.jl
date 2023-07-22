"""
    pcaeigen(X, weights = ones(nro(X)); nlv, scal::Bool = false)
    pcaeigen!(X::Matrix, weights = ones(nro(X)); nlv, scal::Bool = false)
PCA by Eigen factorization.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. principal components (PCs).
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Let us note D the (n, n) diagonal matrix of `weights`
and X the centered matrix in metric D. 
The function minimizes ||X - T * P'||^2  in metric D, by 
computing an Eigen factorization of X' * D * X. 

See `?pcasvd` for examples.
""" 
function pcaeigen(X, weights = ones(nro(X)); nlv, scal::Bool = false)
    pcaeigen!(copy(ensure_mat(X)), weights; nlv = nlv, scal = scal)
end

function pcaeigen!(X::Matrix, weights = ones(nro(X)); nlv, scal::Bool = false)
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
    sqrtw = sqrt.(weights)
    X .= Diagonal(sqrtw) * X
    res = eigen!(Symmetric(X' * X); sortby = x -> -abs(x)) 
    P = res.vectors[:, 1:nlv]
    eig = res.values[1:min(n, p)]
    eig[eig .< 0] .= 0
    sv = sqrt.(eig)
    T = Diagonal(1 ./ sqrtw) * X * P
    Pca(T, P, sv, xmeans, xscales, weights, nothing) 
end
