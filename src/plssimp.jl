"""
    plssimp(X, Y, weights = ones(nro(X)); nlv,
        scal = false)
    plssimp!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        scal = false)
Partial Least Squares Regression (PLSR) with the SIMPLS algorithm (de Jong 1993).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.

**Note:** In this function, scores T (LVs) are not normed, conversely to the original 
algorithm of de Jong (2013)

## References
de Jong, S., 1993. SIMPLS: An alternative approach to partial least squares 
regression. Chemometrics and Intelligent Laboratory Systems 18, 251–263. 
https://doi.org/10.1016/0169-7439(93)85002-X
""" 
function plssimp(X, Y, weights = ones(nro(X)); nlv,
        scal = false)
    plssimp!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv,
        scal = scal)
end

function plssimp!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        scal = false)
    n, p = size(X)
    q = nco(Y)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
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
    D = Diagonal(weights)
    XtY = X' * (D * Y)                   
    # Pre-allocation
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    W = copy(P)
    R = copy(P)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = similar(X, n)   
    zp  = similar(X, p)
    r   = similar(X, p)
    c   = similar(X, q)
    tmp = similar(XtY)
    # End
    # de Jong Chemolab 1993 Table 1 (as fast as Appendix) 
    @inbounds for a = 1:nlv
        if a == 1
            tmp .= XtY
        else
            zP = vcol(P, 1:(a - 1))
            tmp .= XtY .- zP * inv(zP' * zP) * zP' * XtY
        end
        r .= svd!(tmp).U[:, 1] 
        mul!(t, X, r)                 
        dt .= weights .* t            
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
     Plsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing)
end



