"""
    eucl2(X, Y)
Squared Euclidean distances between the rows of `X` and `Y`.
* `X` : Data (n, p).
* `Y` : Data (m, p).

The function returns a matrix (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.

## Examples
```julia
X = rand(5, 3)
Y = rand(2, 3)

eucl2(X, Y)

eucl2(X[1:1, :], Y[1:1, :])

eucl2(X[:, 1], 4)
eucl2(1, 4)
```
"""
function eucl2(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Distances.pairwise(SqEuclidean(), X', Y'; dims = 2)   # pairwise also exportef by StatsBase
end

"""
    mah2(X, Y)
    mah2(X, Y, Sinv)
Squared Mahalanobis distances between the rows of `X` and `Y`.
* `X` : Data (n, p).
* `Y` : Data (m, p).
* `Sinv` : Inverse of a covariance matrix S. If `Sinv` is not given, S is computed as the uncorrected 
    covariance matrix of `X`.

The function returns a matrix (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.

## Examples
```julia
using Jchemo 

X = rand(5, 3)
Y = rand(2, 3)

mah2(X, Y)

S = covm(X)
Sinv = inv(S)
mah2(X, Y, Sinv)
mah2(X[1:1, :], Y[1:1, :], Sinv)

mah2(X[:, 1], 4)
mah2(1, 4, 2.1)
```
"""
function mah2(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    S = covm(X)
    LinearAlgebra.inv!(cholesky!(Hermitian(S)))
    Distances.pairwise(SqMahalanobis(S; skipchecks = true), X', Y'; dims = 2)
end

function mah2(X, Y, Sinv)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Sinv = Hermitian(ensure_mat(Sinv))
    Distances.pairwise(SqMahalanobis(Sinv; skipchecks = true), X', Y'; dims = 2)
end

"""
    mah2chol(X, Y)
    mah2chol(X, Y, Uinv)
Squared Mahalanobis distances (with a Cholesky factorization) between the rows of `X` and `Y`.
* `X` : Data (n, p).
* `Y` : Data (m, p).
* `Uinv` : Inverse of the upper matrix of a Cholesky factorization of a covariance matrix S. 
   If `Uinv` is not given, S is computed as the uncorrected covariance matrix of `X`.

The function returns a matrix (n, m) with:
* i, j = distance between row i of `X` and row j of `Y`.

## Examples
```julia
using LinearAlgebra, StatsBase

X = rand(5, 3)
Y = rand(2, 3)

mah2chol(X, Y)

S = covm(X)
U = cholesky(Hermitian(S)).U 
Uinv = inv(U)
mah2chol(X, Y, Uinv)

mah2chol(X[:, 1], 4)
mah2chol(1, 4, sqrt(2.1))
```
"""
function mah2chol(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)    
    S = covm(X)
    p = nco(S)
    if p == 1
        Uinv = inv(sqrt(S)) 
    else
        Uinv = LinearAlgebra.inv!(cholesky!(Hermitian(S)).U)
    end
    zX = X * Uinv
    zY = Y * Uinv
    eucl2(zX, zY)
end

function mah2chol(X, Y, Uinv)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Uinv = ensure_mat(Uinv)
    zX = X * Uinv
    zY = Y * Uinv
    eucl2(zX, zY)
end

#### Angular and correlation distances (functions not exported)
## All the distances below are scaled to [0, 1]
## Usage:
## ```julia
## n = 1000
## x = rand(n)
## y = rand(n)
## Jchemo.SamDist()(x, y)
## ```julia
struct SamDist <: Distances.Metric end
(::SamDist)(x, y) = acos(1 - Distances.CosineDist()(x, y)) / pi

struct CosDist <: Distances.Metric end                      
(::CosDist)(x, y) = Distances.CosineDist()(x, y) / 2

struct CorDist <: Distances.Metric end                      
(::CorDist)(x, y) = Distances.CorrDist()(x, y) / 2

struct CorDist_b <: Distances.Metric end                            
(::CorDist_b)(x, y) = (1 - corv(x, y)) / 2

## Square-root correlation distance
## max is used since possible negative zeros (floating point issues)
struct CorDist2 <: Distances.Metric end                                
(::CorDist2)(x, y) = sqrt(max(0, Distances.CorrDist()(x, y)) / 2)  

