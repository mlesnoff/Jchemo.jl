"""
    nipals(X; tol = sqrt(eps(1.)), maxit = 200)
    nipals(X, UUt, VVt; tol = sqrt(eps(1.)), maxit = 200)
Nipals to compute the first score and loading vectors of a matrix.
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
X = rand(5, 3)

res = nipals(X)
res.u
svd(X).U[:, 1] 
res.v
svd(X).V[:, 1] 
res.sv
svd(X).S[1] 
res.niter
```
""" 
function nipals(X; tol = sqrt(eps(1.)), maxit = 200)
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
        v ./= norm(v)
        mul!(u, X, v)
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < tol) || (iter > maxit)
            cont = false
        end
    end
    sv = norm(u)
    u .= u / sv
    niter = iter - 1
    (u = u, v, sv, niter)
end

function nipals(X, UUt, VVt; 
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
        v ./= norm(v)
        v .= v .- VVt * v
        mul!(u, X, v)
        u .= u .- UUt * u
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < tol) || (iter > maxit)
            cont = false
        end
    end
    sv = norm(u)
    u .= u / sv
    niter = iter - 1
    (u = u, v, sv, niter)
end


