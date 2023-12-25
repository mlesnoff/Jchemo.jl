"""
    plsnipals(X, Y, weights = ones(nro(X)); nlv,
        scal::Bool = false)
    plsnipals!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        scal::Bool = false)
Partial Least Squares Regression (PLSR) with the Nipals algorithm 
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
* `nlv` : Nb. latent variables (LVs) to consider.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.

In this function, for PLS2 (multivariate Y), the Nipals iterations are replaced by a 
direct computation of the PLS weights (w) by SVD decomposition of matrix X'Y 
(Hoskuldsson 1988 p.213).

See `?plsnipals` for examples.

## References
Hoskuldsson, A., 1988. PLS regression methods. Journal of Chemometrics 2, 211-228.
https://doi.org/10.1002/cem.1180020306

Tenenhaus, M., 1998. La régression PLS: thÃ©orie et pratique. Editions Technip, 
Paris, France.

Wold, S., Sjostrom, M., Eriksson, l., 2001. PLS-regression: a basic tool 
for chemometrics. Chem. Int. Lab. Syst., 58, 109-130.
""" 
function plsnipals(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plsnipals(X, Y, weights; kwargs...)
end

function plsnipals(X, Y, weights::Weight; kwargs...)
    plsnipals!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; kwargs...)
end

function plsnipals!(X::Matrix, Y::Matrix, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs)
    Q = eltype(X)
    n, p = size(X)
    q = nco(Y)
    nlv = min(par.nlv, n, p)
    D = Diagonal(weights.w)
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
    # Pre-allocation
    XtY = similar(X, p, q)
    T = similar(X, n, nlv)
    W = similar(X, p, nlv)
    P = copy(W)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = similar(X, n)   
    w = similar(X, p)
    zp  = copy(w)
    c   = similar(X, q)
    # End
    @inbounds for a = 1:nlv
        XtY .= X' * D * Y
        if q == 1
            w .= vec(XtY)
            w ./= norm(w)
        else
            w .= svd!(XtY).U[:, 1]
        end
        mul!(t, X, w)
        dt .= weights.w .* t
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
    Plsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, 
        yscales, weights, nothing, kwargs, par)
end

