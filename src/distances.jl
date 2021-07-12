"""
    euclsq(X, Y)
Compute the squared Euclidean distances 
between the observations (rows) of `X` and `Y`.
* `X` : Data.
* `Y` : Data.

When `X` (n, p) and `Y` (m, p), it returns an object (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.
"""
function euclsq(X, Y)
    Xt = ensure_mat(X')
    Yt = ensure_mat(Y')
    Distances.pairwise(SqEuclidean(), Xt, Yt, dims = 2)
end

"""
    mahsq(X, Y)
    mahsq(X, Y, Sinv)
Compute the squared Mahalanobis distances 
between the observations (rows) of `X` and `Y`.
* `X` : Data.
* `Y` : Data.
* `Sinv` : Inverse of a covariance matrix S.
    If not given, this is the uncorrected covariance matrix of `X`.

When `X` (n, p) and `Y` (m, p), it returns an object (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.
"""
function mahsq(X, Y)
    X = ensure_mat(X)
    Xt = ensure_mat(X')
    Yt = ensure_mat(Y')
    S = Statistics.cov(X, corrected = false)
    Sinv = inv(S) 
    Distances.pairwise(SqMahalanobis(Sinv), Xt, Yt, dims = 2)
end

function mahsq(X, Y, Sinv)
    Xt = ensure_mat(X')
    Yt = ensure_mat(Y')
    Sinv = ensure_mat(Sinv)
    Distances.pairwise(SqMahalanobis(Sinv), Xt, Yt, dims = 2)
end

"""
    mahsqchol(X, Y)
    mahsqchol(X, Y, U)
Compute the squared Mahalanobis distances (with a Cholesky factorization)
between the observations (rows) of `X` and `Y`.
* `X` : Data.
* `Y` : Data.
* `U` : Cholesky factorization of a covariance matrix S.
    If not given, the factorization is done on S, the uncorrected covariance matrix of `X`.

When `X` (n, p) and `Y` (m, p), it returns an object (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.
"""
function mahsqchol(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)    
    p = size(X, 2)
    S = Statistics.cov(X, corrected = false)
    if p == 1
        U = sqrt(S) 
    else
        U = cholesky(Hermitian(S)).U
    end
    Uinv = inv(U)
    zX = X * Uinv
    zY = Y * Uinv
    euclsq(zX, zY)
end

function mahsqchol(X, Y, U)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    U = ensure_mat(U)
    Uinv = inv(U)
    zX = X * Uinv
    zY = Y * Uinv
    euclsq(zX, zY)
end





