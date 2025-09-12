"""
    euclsq(X, Y)
Squared Euclidean distances between the rows of `X` and `Y`.
* `X` : Data (n, p).
* `Y` : Data (m, p).

For `X`(n, p) and `Y` (m, p), the function returns 
an object (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.

## Examples
```julia
X = rand(5, 3)
Y = rand(2, 3)

euclsq(X, Y)

euclsq(X[1:1, :], Y[1:1, :])

euclsq(X[:, 1], 4)
euclsq(1, 4)
```
"""
function euclsq(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Distances.pairwise(SqEuclidean(), X', Y', dims = 2)
end

"""
    mahsq(X, Y)
    mahsq(X, Y, Sinv)
Squared Mahalanobis distances between 
    the rows of `X` and `Y`.
* `X` : Data (n, p).
* `Y` : Data (m, p).
* `Sinv` : Inverse of a covariance matrix S.
    If not given, S is computed as the uncorrected 
    covariance matrix of `X`.

When `X` and `Y` are (n, p) and (m, p), repectively, it returns 
an object (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.

## Examples
```julia
using Jchemo 

X = rand(5, 3)
Y = rand(2, 3)

mahsq(X, Y)

S = covm(X)
Sinv = inv(S)
mahsq(X, Y, Sinv)
mahsq(X[1:1, :], Y[1:1, :], Sinv)

mahsq(X[:, 1], 4)
mahsq(1, 4, 2.1)
```
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
    Distances.pairwise(SqMahalanobis(Sinv; skipchecks = true), X', Y', dims = 2)
end

"""
    mahsqchol(X, Y)
    mahsqchol(X, Y, Uinv)
Compute the squared Mahalanobis distances (with a Cholesky factorization)
between the observations (rows) of `X` and `Y`.
* `X` : Data (n, p).
* `Y` : Data (m, p).
* `Uinv` : Inverse of the upper matrix of a Cholesky factorization 
    of a covariance matrix S.
    If not given, the factorization is done on S, 
    the uncorrected covariance matrix of `X`.

When `X` and `Y` are (n, p) and (m, p), repectively, it returns 
an object (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.

## Examples
```julia
using LinearAlgebra, StatsBase

X = rand(5, 3)
Y = rand(2, 3)

mahsqchol(X, Y)

S = cov(X, corrected = false)
U = cholesky(Hermitian(S)).U 
Uinv = inv(U)
mahsqchol(X, Y, Uinv)

mahsqchol(X[:, 1], 4)
mahsqchol(1, 4, sqrt(2.1))
```
"""
function mahsqchol(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)    
    p = nco(X)
    S = Statistics.cov(X, 
        corrected = false)
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

#### Angular and correlation distances (not exported)
## All the distances below are scaled to [0, 1]
## Usage:
## n = 1000
## x = rand(n)
## y = rand(n)
## Jchemo.SamDist()(x, y)
struct SamDist <: Distances.Metric end
(::SamDist)(a, b) = acos(1 - Distances.CosineDist()(a, b)) / pi

struct CosDist <: Distances.Metric end                      
(::CosDist)(a, b) = Distances.CosineDist()(a, b) / 2

struct CorDist <: Distances.Metric end                      
(::CorDist)(a, b) = Distances.CorrDist()(a, b) / 2

struct CorDist_b <: Distances.Metric end                            
(::CorDist_b)(a, b) = (1 - corv(a, b)) / 2

## Square-root correlation distance
## max is used since possible negative zeros (floating point issues)
struct CorDist_sqr <: Distances.Metric end                                
(::CorDist_sqr)(a, b) = sqrt(max(0, Distances.CorrDist()(a, b)) / 2)  

