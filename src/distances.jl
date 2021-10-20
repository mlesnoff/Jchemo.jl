"""
    euclsq(X, Y)
Squared Euclidean distances 
between the rows of `X` and `Y`.
* `X` : Data.
* `Y` : Data.

When `X` (n, p) and `Y` (m, p), it returns an object (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.
"""
function euclsq(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Distances.pairwise(SqEuclidean(), X', Y', dims = 2)
end

"""
    mahsq(X, Y)
    mahsq(X, Y, Sinv)
Squared Mahalanobis distances 
between the rows of `X` and `Y`.
* `X` : Data.
* `Y` : Data.
* `Sinv` : Inverse of a covariance matrix S.
    If not given, this is the uncorrected covariance matrix of `X`.

For `X` (n, p) and `Y` (m, p), it returns an object (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.
"""
function mahsq(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    S = Statistics.cov(X, corrected = false)
    LinearAlgebra.inv!(cholesky!(Hermitian(S)))
    Distances.pairwise(SqMahalanobis(S), X', Y', dims = 2)
end

function mahsq(X, Y, Sinv)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Sinv = ensure_mat(Sinv)
    Distances.pairwise(SqMahalanobis(Sinv), X', Y', dims = 2)
end

"""
    mahsqchol(X, Y)
    mahsqchol(X, Y, Uinv)
Compute the squared Mahalanobis distances (with a Cholesky factorization)
between the observations (rows) of `X` and `Y`.
* `X` : Data.
* `Y` : Data.
* `Uinv` : Inverse of the upper matrix of a Cholesky factorization 
    of a covariance matrix S.
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
        Uinv = inv(sqrt(S)) 
    else
        Uinv = LinearAlgebra.inv!(cholesky!(Hermitian(S)).U)
    end
    zX = X * Uinv
    zY = Y * Uinv
    euclsq(zX, zY)
end

function mahsqchol(X, Y, Uinv)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Uinv = ensure_mat(Uinv)
    zX = X * Uinv
    zY = Y * Uinv
    euclsq(zX, zY)
end





