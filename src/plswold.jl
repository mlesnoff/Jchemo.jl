"""
    plswold(X, Y, weights = ones(nro(X)); nlv,
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    plswold!(X, Y, weights = ones(nro(X)); nlv,
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
Partial Least Squares Regression (PLSR) with the Wold algorithm 
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. latent variables (LVs) to consider.
* `tol` : Tolerance for the Nipals algorithm.
* `maxit` : Maximum number of iterations for the Nipals algorithm.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.

Wold Nipals PLSR algorithm: Tenenhaus 1998 p.204.
    
See `?plskern` for examples.

## References
Tenenhaus, M., 1998. La régression PLS: thÃ©orie et pratique. Editions Technip, 
Paris, France.

Wold, S., Ruhe, A., Wold, H., Dunn, III, W.J., 1984. The Collinearity Problem in 
Linear Regression. The Partial Least Squares (PLS). Approach to 
Generalized Inverses. SIAM Journal on Scientific and Statistical Computing 5, 735–743. 
https://doi.org/10.1137/0905052
""" 
function plswold(X, Y, weights = ones(nro(X)); nlv,
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    plswold!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv,
        tol = tol, maxit = maxit, scal = scal)
end

function plswold!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    n, p = size(X)
    q = nco(Y)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    sqrtw = sqrt.(weights)
    D = Diagonal(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(p)
    yscales = ones(q)
    if scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        cscale!(X, xmeans, xscales)
        cscale!(Y, ymeans, yscales)
    else
        center!(X, xmeans)
        center!(Y, ymeans)
    end
    # Row metric
    X .= sqrtw .* X
    Y .= sqrtw .* Y
    # Pre-allocation
    Tx = similar(X, n, nlv)
    Wx = similar(X, p, nlv)
    Wytild = similar(X, q, nlv)
    Px = copy(Wx)
    TTx = similar(X, nlv)
    tx   = similar(X, n)
    ty = copy(tx)
    wx  = similar(X, p)
    wxtild = copy(wx)    
    wy  = similar(X, q)
    wytild = copy(wy)    # = ctild 
    px   = copy(wx)
    py   = copy(wy)
    niter = zeros(nlv)
    # End
    @inbounds for a = 1:nlv
        tx .= X[:, 1]
        ty .= Y[:, 1]
        cont = true
        iter = 1
        wx .= rand(p)
        while cont
            w0 = copy(wx)
            wxtild .= X' * ty / dot(ty, ty) 
            wx .= wxtild / norm(wxtild)
            tx .= X * wx
            wytild = Y' * tx / dot(tx, tx)
            wy .= wytild / norm(wytild)
            ty .= Y * wy
            dif = sum((wx .- w0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        ttx = dot(tx, tx)
        mul!(px, X', tx)
        px ./= ttx
        tty = dot(ty, ty)
        mul!(py, Y', ty)
        py ./= tty
        # Deflation
        X .-= tx * px'
        Y .-= tx * wytild'
        # End         
        Tx[:, a] .= tx
        Wx[:, a] .= wx
        Px[:, a] .= px
        Wytild[:, a] .= wytild
        TTx[a] = ttx
     end
     Tx .= (1 ./ sqrtw) .* Tx
     Rx = Wx * inv(Px' * Wx)
     #Plsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing)
     Plsr(Tx, Px, Rx, Wx, Wytild, TTx, 
        xmeans, xscales, ymeans, yscales, weights, niter)
end
