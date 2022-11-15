"""
    plsnipals(X, Y, weights = ones(nro(X)); nlv,
        scal = false)
    plsnipals!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        scal = false)
Partial Least Squares Regression (PLSR) with the NIPALS algorithm 
(e.g. Tenenhaus 1998, Wold 2002).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. latent variables (LVs) to consider.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.

**Note:** In this function, for PLS2, the NIPALS iterations are replaced by a 
direct computation of the PLS weights (w) by SVD decomposition of matrix X'Y 
(Hoskuldsson 1988 p.213).

See `?plskern` for examples.

## References
Hoskuldsson, A., 1988. PLS regression methods. Journal of Chemometrics 2, 211-228.
https://doi.org/10.1002/cem.1180020306

Tenenhaus, M., 1998. La régression PLS: thÃ©orie et pratique. Editions Technip, 
Paris, France.

Wold, S., Sjostrom, M., Eriksson, l., 2001. PLS-regression: a basic tool 
for chemometrics. Chem. Int. Lab. Syst., 58, 109-130.
""" 
function plsnipals(X, Y, weights = ones(nro(X)); nlv,
        scal = false)
    plsnipals!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv,
        scal = scal)
end

function plsnipals!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        scal = false)
    n, p = size(X)
    q = nco(Y)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
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
    # Pre-allocation
    XtY = similar(X, p, q)
    T = similar(X, n, nlv)
    W  = similar(X, p, nlv)
    P  = copy(W)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = similar(X, n)   
    w = similar(X, p)
    zp  = copy(w)
    c   = similar(X, q)
    # End
    @inbounds for a = 1:nlv
        XtY .= X' * (D * Y)
        if q == 1
            w .= vec(XtY)
            w ./= norm(w)
        else
            w .= svd!(XtY).U[:, 1]
        end
        mul!(t, X, w)
        dt .= weights .* t
        tt = dot(t, dt)
        mul!(zp, X', dt)
        zp ./= tt
        mul!(c, Y', dt)
        c ./= tt                      
        # deflation with respect to t (asymetric PLS)
        X .-= t * zp'
        Y .-= t * c'
        # end
        P[:, a] .= zp  
        T[:, a] .= t
        W[:, a] .= w
        C[:, a] .= c
        TT[a] = tt
     end
     R = W * inv(P' * W)
     Plsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing)
end

