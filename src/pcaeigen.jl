"""
    pcaeigen(X, weights = ones(size(X, 1)); nlv)
PCA by Eigen factorization.
* `X` : X-data.
* `weights` : Weights of the observations.
* `nlv` : Nb. principal components (PCs).
    
Let us note D the (n, n) diagonal matrix of `weights` (internally normalized to sum to 1)
and X the centered matrix in metric D (`X` is internally centered). 
The function minimizes ||X - T * P'||^2  in metric D, by 
computing an Eigen factorization of X' * D * X. 
""" 
function pcaeigen(X, weights = ones(size(X, 1)); nlv)
    pcaeigen!(copy(X), weights; nlv = nlv)
end

function pcaeigen!(X, weights = ones(size(X, 1)); nlv)
    X = ensure_mat(X)
    n, p = size(X)
    nlv = min(nlv, n, p)
    weights = mweights(weights)
    sqrtw = sqrt.(weights)
    xmeans = colmeans(X, weights) 
    center!(X, xmeans)
    X .= Diagonal(sqrtw) * X
    res = eigen!(Symmetric(X' * X); sortby = x -> -abs(x)) 
    P = res.vectors[:, 1:nlv]
    eig = res.values[1:min(n, p)]
    eig[eig .< 0] .= 0
    sv = sqrt.(eig)
    T = Diagonal(1 ./ sqrtw) * X * P
    Pca(T, P, sv, xmeans, weights, nothing, nothing) 
end

"""
    pcaeigenk(X, weights = ones(size(X, 1)); nlv)
PCA by Eigen factorization of the kernel form (XX') for wide matrices.
* `X` : X-data.
* `weights` : Weights of the observations.
* `nlv` : Nb. principal components (PCs).

Let us note D the (n, n) diagonal matrix of `weights` (internally normalized to sum to 1)
and X the centered matrix in metric D (`X` is internally centered). 
The function minimizes ||X - T * P'||^2  in metric D, by 
computing an Eigen factorization of D^(1/2) * X * X' D^(1/2).

This is the "kernel cross-product" version of the PCA algorithm (e.g. Wu et al. 1997). 
For wide matrices (n << p, where p is the nb. columns) and n not too large, 
this algorithm can be much faster than the others.

## References
Wu, W., Massart, D.L., de Jong, S., 1997. The kernel PCA algorithms for wide data. Part I: Theory and algorithms. 
Chemometrics and Intelligent Laboratory Systems 36, 165-172. https://doi.org/10.1016/S0169-7439(97)00010-5
""" 
function pcaeigenk(X, weights = ones(size(X, 1)); nlv)
    pcaeigenk!(copy(X), weights; nlv = nlv)
end

function pcaeigenk!(X, weights = ones(size(X, 1)); nlv)
    X = ensure_mat(X)
    n, p = size(X)
    nlv = min(nlv, n, p)
    weights = mweights(weights)
    sqrtw = sqrt.(weights)
    xmeans = colmeans(X, weights) 
    center!(X, xmeans)
    zX = Diagonal(sqrtw) * X
    res = eigen!(Symmetric(zX * zX'); sortby = x -> -abs(x))
    eig = res.values[1:min(n, p)]
    eig[eig .< 0] .= 0
    sv = sqrt.(eig)
    P = zX' * scale(res.vectors[:, 1:nlv], sv[1:nlv])
    T = X * P
    Pca(T, P, sv, xmeans, weights, nothing, nothing) 
end


