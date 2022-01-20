"""
    nipals(X, weights = ones(size(X, 1)); tol = sqrt(eps(1.)), 
        maxit = 200)
NIPALS to compute the first score and loading vectors of a matrix.
* `X` : X-data (n, p).
* `weights` : Weights of the observations.
* `tol` : Tolerance value for stopping the iterations.
* `maxit` : Maximum nb. iterations?

Noting D the (n, n) diagonal matrix of weights of the observations (rows of X), 
the function finds {t, w} = argmin(||D^(1/2) * X - t'w||), with the constraint ||w|| = 1.

Vector `weights` is internally normalized to sum to 1.

## References
Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, 
Paris, France.
""" 
function nipals(X, weights = ones(size(X, 1)); tol = sqrt(eps(1.)), 
        maxit = 200)
    X = ensure_mat(X)
    n, p = size(X)
    weights = mweights(weights)
    # Pre-allocation
    t = similar(X, n)
    dt = copy(t)
    w = similar(X, p)   
    # End
    t .= X[:, findmax(colvars(X, weights))[2]]
    cont = true
    iter = 1
    while cont
        t0 = copy(t)
        dt .= weights .* t          
        mul!(w, X', dt)
        w ./= norm(w)
        mul!(t, X, w)
        dif = sum((t .- t0).^2)
        iter = iter + 1
        if (dif < tol) || (iter > maxit)
            cont = false
        end
    end
    sv = norm(t, weights)
    niter = iter - 1
    (t = t, w, sv, niter)
end
