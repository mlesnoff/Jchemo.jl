"""
    nipalsmiss(X; kwargs...)
    nipalsmiss(X, UUt, VVt; kwargs...)
Nipals to compute the first score and loading vectors 
    of a matrix with missing data.
* `X` : X-data (n, p).
* `UUt` : Matrix (n, n) for Gram-Schmidt orthogonalization.
* `VVt` : Matrix (p, p) for Gram-Schmidt orthogonalization.
Keyword arguments:
* `tol` : Tolerance value for stopping 
    the iterations.
* `maxit` : Maximum nb. of iterations.

See function `nipals`. 

## References
K.R. Gabriel, S. Zamir, Lower rank approximation of 
matrices by least squares with any choice of weights, 
Technometrics 21 (1979) 489–498.

Wright, K., 2018. Package nipals: Principal Components 
Analysis using NIPALS with Gram-Schmidt Orthogonalization. 
https://cran.r-project.org/

## Examples
```julia
using Jchemo 

X = [1. 2 missing 4 ; 4 missing 6 7 ; 
    missing 5 6 13 ; missing 18 7 6 ; 
    12 missing 28 7] 

res = nipalsmiss(X)
res.niter
res.sv
res.v
res.u
```
""" 
function nipalsmiss(X; kwargs...)
    par = recovkw(ParNipals, kwargs).par
    X = ensure_mat(X)
    n, p = size(X)
    s = ismissing.(X)
    st = ismissing.(X')
    X0 = copy(X)
    X0[s] .= 0
    X0t = X0'
    zU = similar(X0, n, p)
    zV = similar(X0, p, n)
    u = X0[:, argmax(colsumskip(abs.(X)))]
    u0 = similar(X0, n)
    v = similar(X0, p) 
    cont = true
    iter = 1
    while cont
        u0 .= copy(u)
        zU .= reshape(repeat(u.^2, p), n, p)
        zU[s] .= 0
        mul!(v, X0t, u)
        v ./= colsum(zU)
        v ./= normv(v)
        zV .= reshape(repeat(v.^2, n), p, n)
        zV[st] .= 0
        mul!(u, X0, v)
        u ./= colsum(zV)
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

function nipalsmiss(X, UUt, VVt; kwargs...)
    par = recovkw(ParNipals, kwargs).par
    X = ensure_mat(X)
    n, p = size(X)
    s = ismissing.(X)
    st = ismissing.(X')
    X0 = copy(X)
    X0[s] .= 0
    X0t = X0'
    zU = similar(X0, n, p)
    zV = similar(X0, p, n)
    u = X0[:, argmax(colsumskip(abs.(X)))]
    u0 = similar(X0, n)
    v = similar(X0, p) 
    cont = true
    iter = 1
    while cont
        u0 .= copy(u)
        zU .= reshape(repeat(u.^2, p), n, p)
        zU[s] .= 0
        mul!(v, X0t, u)
        v ./= colsum(zU)
        v .-= VVt * v
        v ./= normv(v)
        zV .= reshape(repeat(v.^2, n), p, n)
        zV[st] .= 0
        mul!(u, X0, v)
        u ./= colsum(zV)
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

