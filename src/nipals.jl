"""
    nipals(X; tol = sqrt(eps(1.)), maxit = 200)
Nipals to compute the first score and loading vectors of a matrix.
* `X` : X-data (n, p).
* `tol` : Tolerance value for stopping the iterations.
* `maxit` : Maximum nb. iterations.

The function finds {u, v, s} = argmin(||X - u * s * v'||), with the constraints 
||u|| = ||v|| = 1.

X ~ u * s * v', where:

* u, v : left and right singular vectors (scores and loadings, repectively)
* s : singular value.
    
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
    u = X[:, findmax(colnorm(X))[2]]
    v = similar(X, p)   
    cont = true
    iter = 1
    while cont
        u0 = copy(u)      
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
    n = size(X, 1)
    (u = u, v, sv, niter)
end






