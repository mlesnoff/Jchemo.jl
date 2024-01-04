"""
    plssimp(; kwargs...)
    plssimp(X, Y; kwargs...)
    plssimp(X, Y, weights::Weight; 
        kwargs...)
    plssimp!(X::Matrix, Y::Matrix, weights::Weight; 
        kwargs...)
Partial Least Squares Regression (PLSR) with the 
    SIMPLS algorithm (de Jong 1993).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.

**Note:** In this function, scores T (LVs) are not normed, 
conversely to the original algorithm of de Jong (2013).

See function `plskern` for examples.

## References
de Jong, S., 1993. SIMPLS: An alternative approach to 
partial least squares regression. Chemometrics and Intelligent 
Laboratory Systems 18, 251–263. 
https://doi.org/10.1016/0169-7439(93)85002-X
""" 
function plssimp(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plssimp(X, Y, weights; kwargs...)
end

function plssimp(X, Y, weights::Weight; kwargs...)
    plssimp!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; kwargs...)
end

function plssimp!(X::Matrix, Y::Matrix, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs)
    Q = eltype(X)
    n, p = size(X)
    q = nco(Y)
    nlv = min(par.nlv, n, p)
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
    D = Diagonal(weights.w)
    XtY = X' * (D * Y)   
    ## Pre-allocation
    T = similar(X, n, nlv)
    W = similar(X, p, nlv)
    P = copy(W)
    R = copy(P)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = similar(X, n)   
    zp  = similar(X, p)
    r   = similar(X, p)
    c   = similar(X, q)
    tmpXtY = similar(XtY)
    # End
    # de Jong Chemolab 1993 Table 1 (as fast as Appendix) 
    @inbounds for a = 1:nlv
        if a == 1
            tmpXtY .= XtY
        else
            Pa = vcol(P, 1:(a - 1))
            tmpXtY .= XtY .- Pa * inv(Pa' * Pa) * Pa' * XtY
        end
        r .= svd!(tmpXtY).U[:, 1] 
        mul!(t, X, r)                 
        dt .= weights.w .* t            
        tt = dot(t, dt)
        mul!(c, XtY', r)
        c ./= tt                      
        mul!(zp, X', dt)
        P[:, a] .= zp ./ tt
        T[:, a] .= t
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
    end
    #B = R * inv(T' * D * T) * T' * D * Y
    # W does not exist in SIMPLS
    # Below it is filled by R (for vip)
    Plsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, 
        yscales, weights, nothing, kwargs, par)
end



