"""
    nipalsmiss(X; tol = sqrt(eps(1.)), maxit = 200)
    nipalsmiss(X, UUt, VVt; tol = sqrt(eps(1.)), maxit = 200)
Nipals to compute the first score and loading vectors of a matrix
    with missing data.
* `X` : X-data (n, p).
* `UUt` : Matrix (n, n) for Gram-Schmidt orthogonalization.
* `VVt` : Matrix (p, p) for Gram-Schmidt orthogonalization.
* `tol` : Tolerance value for stopping the iterations.
* `maxit` : Maximum nb. iterations.

The function finds {u, v, sv} = argmin(||X - u * sv * v'||), with the constraints 
||u|| = ||v|| = 1.

X ~ u * sv * v', where:

* u : left singular vector (u * sv = scores)
* v : right singular vector (loadings)
* sv : singular value.

When NIPALS is used sequentially on deflated matrices, vectors u and v 
can loose orthogonality due to rounding errors effects. Orthogonality can
be rebuilt from the Gram-Schmidt method (use of `UUt` and `VVt`). 

## References
Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, 
Paris, France.

## Examples
```julia
X = [1. 2 missing 4 ; 4 missing 6 7 ; 
    missing 5 6 13 ; missing 18 7 6 ; 
    12 missing 28 7] 

res = nipalsmiss(X)
res.u
svd(X).U[:, 1] 
res.v
svd(X).V[:, 1] 
res.sv
svd(X).S[1] 
res.niter
```
""" 
function nipalsmiss(X; tol = sqrt(eps(1.)), maxit = 200)
    X = ensure_mat(X)
    n, p = size(X)
    X0 = copy(X)
    s = ismissing.(X0)
    ts = ismissing.(X0')
    X0[s] .= 0
    X0t = X0'
    u = X0[:, argmax(colsumskip(abs.(X)))]
    u0 = copy(u)
    zU = similar(X, n, p)
    zV = similar(X, p, n)
    v = similar(X, p) 
    cont = true
    iter = 1
    while cont
        u0 .= copy(u)
        zU .= reshape(repeat(u.^2, p), n, p)
        zU[s] .= 0
        mul!(v, X0t, u)
        v ./= colsum(zU)
        v ./= norm(v)
        zV .= reshape(repeat(v.^2, n), p, n)
        zV[ts] .= 0
        mul!(u, X0, v)
        u ./= colsum(zV)
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < tol) || (iter > maxit)
            cont = false
        end
    end
    sv = norm(u)
    u ./= sv
    niter = iter - 1
    (u = u, v, sv, niter)
end

function nipalsmiss(X, UUt, VVt; 
        tol = sqrt(eps(1.)), maxit = 200)
    X = ensure_mat(X)
    p = nco(X)
    u = X[:, argmax(colnorm(X))]
    u0 = copy(u)
    v = similar(X, p)   
    cont = true
    iter = 1
    while cont
        u0 .= copy(u)      
        mul!(v, X', u)
        v .= v .- VVt * v
        v ./= norm(v)
        mul!(u, X, v)
        u .= u .- UUt * u
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < tol) || (iter > maxit)
            cont = false
        end
    end
    sv = norm(u)
    u ./= sv
    niter = iter - 1
    (u = u, v, sv, niter)
end


