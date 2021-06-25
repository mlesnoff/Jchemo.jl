"""
    pcaeigen(X, weights = ones(size(X, 1)); nlv)
PCA by Eigen decomposition.
* `X` : matrix (n, p).
* `weights` : vector (n,).
* `nlv` : Nb. principal components (PCs).
    
Noting D a (n, n) diagonal matrix of weights for the observations (rows of `X`),
the function does an Eigen factorization of X' * D * X, using LinearAlgebra.eigen.
    
`X` is internally centered. 

The in-place version modifies externally `X`. 
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
    zX = Diagonal(sqrtw) * X
    res = eigen!(Symmetric(zX' * zX); sortby = x -> -abs(x)) 
    P = res.vectors[:, 1:nlv]
    eig = res.values[1:min(n, p)]
    eig[eig .< 0] .= 0
    sv = sqrt.(eig)
    T = X * P
    Pca(T, P, sv, xmeans, weights, nothing, nothing)
end

"""
    pcaeigenk(X, weights = ones(size(X, 1)); nlv)
PCA by Eigen decomposition: kernel version for wide matrices.
* `X` : matrix (n, p).
* `weights` : vector (n,).
* `nlv` : Nb. principal components (PCs).

Noting D a (n, n) diagonal matrix of weights for the observations (rows of X),
the function does an Eigen factorization of D^(1/2) * X * X' D^(1/2), using LinearAlgebra.eigen.
This is the "kernel cross-product trick" version of the PCA algorithm (Wu et al. 1997). 
For wide matrices (n << p) and n not too large, this algorithm can be much faster than the others.

`X` is internally centered. 

The in-place version modifies externally `X`. 

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
    zV = scale(res.vectors[:, 1:nlv], sv[1:nlv])
    P = zX' * zV
    T = X * P
    Pca(T, P, sv, xmeans, weights, nothing, nothing)
end


