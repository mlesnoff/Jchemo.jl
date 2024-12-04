"""
    nipals(X; kwargs...)
    nipals(X, UUt, VVt; kwargs...)
Nipals to compute the first score and loading vectors of a matrix.
* `X` : X-data (n, p).
* `UUt` : Matrix (n, n) for Gram-Schmidt orthogonalization.
* `VVt` : Matrix (p, p) for Gram-Schmidt orthogonalization.
Keyword arguments:
* `tol` : Tolerance value for stopping 
    the iterations.
* `maxit` : Maximum nb. of iterations.

The function finds:
* {u, v, sv} = argmin(||X - u * sv * v'||)
with the constraints:
* ||u|| = ||v|| = 1
using the alternating least squares algorithm to compute 
SVD (Gabriel & Zalir 1979).

At the end, X ~ u * sv * v', where:
* u : left singular vector (u * sv = scores)
* v : right singular vector (loadings)
* sv : singular value.

When NIPALS is used on sequentially deflated matrices, 
vectors u and v can loose orthogonality due to accumulation 
of rounding errors. Orthogonality can be rebuilt from the 
Gram-Schmidt method (arguments `UUt` and `VVt`). 

## References
K.R. Gabriel, S. Zamir, Lower rank approximation of matrices
by least squares with any choice of weights, 
Technometrics 21 (1979) 489–498.

## Examples
```julia
using Jchemo, LinearAlgebra

X = rand(5, 3)

res = nipals(X)
res.niter
res.sv
svd(X).S[1] 
res.v
svd(X).V[:, 1] 
res.u
svd(X).U[:, 1] 
```
""" 
function nipals(X; kwargs...)
    par = recovkw(ParNipals, kwargs).par
    X = ensure_mat(X)
    n, p = size(X)
    u = X[:, argmax(colnorm(X))]
    u0 = similar(X, n)
    v = similar(X, p)   
    cont = true
    iter = 1
    while cont
        u0 .= copy(u)      
        mul!(v, X', u)
        v ./= normv(v)
        mul!(u, X, v)
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < par.tol) || (iter > par.maxit)
            cont = false
        end
    end
    sv = normv(u)
    u ./= sv
    niter = iter - 1
    (u = u, v, sv, niter)
end

function nipals(X, UUt, VVt; kwargs...)
    par = recovkw(ParNipals, kwargs).par
    X = ensure_mat(X)
    n, p = size(X)
    u = X[:, argmax(colnorm(X))]
    u0 = similar(X, n)
    v = similar(X, p)   
    cont = true
    iter = 1
    while cont
        u0 .= copy(u)      
        mul!(v, X', u)
        v .= v .- VVt * v
        v ./= normv(v)
        mul!(u, X, v)
        u .= u .- UUt * u
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < par.tol) || (iter > par.maxit)
            cont = false
        end
    end
    sv = normv(u)
    u ./= sv
    niter = iter - 1
    (u = u, v, sv, niter)
end


