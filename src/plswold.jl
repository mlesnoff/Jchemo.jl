"""
    plswold(X, Y; kwargs...)
    plswold(X, Y, weights::Weight; kwargs...)
    plswold!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Partial Least Squares Regression (PLSR) with the Wold algorithm 
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `tol` : Tolerance for the Nipals algorithm.
* `maxit` : Maximum number of iterations for the Nipals algorithm.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.

Wold Nipals PLSR algorithm: Tenenhaus 1998 p.204.
    
See function `plskern` for examples.

## References
Tenenhaus, M., 1998. La régression PLS: thÃ©orie et pratique. 
Editions Technip, Paris, France.

Wold, S., Ruhe, A., Wold, H., Dunn, III, W.J., 1984. The 
Collinearity Problem in Linear Regression. The Partial Least 
Squares (PLS). Approach to Generalized Inverses. SIAM Journal on 
Scientific and Statistical Computing 5, 735–743. 
https://doi.org/10.1137/0905052
""" 
function plswold(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plswold(X, Y, weights; kwargs...)
end

function plswold(X, Y, weights::Weight; kwargs...)
    plswold!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plswold!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPlsr, kwargs).par
    Q = eltype(X)
    n, p = size(X)
    q = nco(Y)
    nlv = min(par.nlv, n, p)
    sqrtw = sqrt.(weights.w)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        fcscale!(X, xmeans, xscales)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
    end
    # Row metric
    X .= sqrtw .* X
    Y .= sqrtw .* Y
    ## Pre-allocation
    Tx = similar(X, n, nlv)
    Wx = similar(X, p, nlv)
    Wytild = similar(X, q, nlv)
    Px = copy(Wx)
    TTx = similar(X, nlv)
    tx   = similar(X, n)
    ty = copy(tx)
    wx  = similar(X, p)
    wy  = similar(X, q)
    wytild = copy(wy)     
    px   = copy(wx)
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
            wx .= X' * ty / dot(ty, ty)    
            wx ./= norm(wx)
            tx .= X * wx
            wytild = Y' * tx / dot(tx, tx)    # = ctild ==> output "C"
            wy .= wytild / norm(wytild)
            ty .= Y * wy
            dif = sum((wx .- w0).^2)
            iter = iter + 1
            if (dif < par.tol) || (iter > par.maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        ttx = dot(tx, tx)
        mul!(px, X', tx)
        px ./= ttx
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
    Plsr(Tx, Px, Rx, Wx, Wytild, TTx, xmeans, xscales, ymeans, yscales, weights, niter, par)
end
