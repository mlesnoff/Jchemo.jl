"""
    euclsq(X, Y)
Squared Euclidean distances between the rows
of a matrix X (n, p) and the rows of a matrix Y (m, p).
Return a n x m matrix: i, j = distance between row i of X and row j of Y.
X and Y must have the same number of columns.
X and Y can be vectors or scalars if dimensions are consistent.
"""
function euclsq(X, Y)
    Xt = ensure_mat(X')
    Yt = ensure_mat(Y')
    Distances.pairwise(SqEuclidean(), Xt, Yt, dims = 2)
end

"""
    mahsq(X, Y)
    mahsq(X, Y, Sinv)
Squared Mahalanobis distances between the rows
of a matrix X (n, p) and the rows of a matrix Y (m, p).
Return a n x m matrix: i, j = distance between row i of X and row j of Y.
X and Y must have the same number of columns.
X and Y can be vectors or scalars if dimensions are consistent.
- Sinv: Inverse of a covariance matrix S (p, p)
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
Squared Mahalanobis distances between the rows
of a matrix X (n, p) and the rows of a matrix Y (m, p).
Return a n x m matrix: i, j = distance between row i of X and row j of Y.
X and Y must have the same number of columns.
X and Y can be vectors or scalars if dimensions are consistent.
- U: Cholesky decomposition of the covariance matrix S (p, p) 
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





