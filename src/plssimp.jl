"""
    plssimp(; kwargs...)
    plssimp(X, Y; kwargs...)
    plssimp(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    plssimp!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
Partial Least Squares Regression (PLSR) with the SIMPLS algorithm (de Jong 1993).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected 
    standard deviation.

**Note:** In this function, scores T (LVs) are not normed, conversely to the original algorithm of de Jong (2013).

See function `plskern` for examples.

## References
de Jong, S., 1993. SIMPLS: An alternative approach to partial least squares regression. Chemometrics and Intelligent 
Laboratory Systems 18, 251–263. https://doi.org/10.1016/0169-7439(93)85002-X
""" 
plssimp(; kwargs...) = JchemoModel(plssimp, nothing, kwargs)

function plssimp(X, Y; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    plssimp(X, Y, weights; kwargs...)
end

function plssimp(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    plssimp!(copy(X), copy(Y), weights; kwargs...)
end

function plssimp!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = recovkw(ParPlsr, kwargs).par
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, par.nlv)
    par.nlv = nlv
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
        yscales .= colscal(Y, weights)
        fcscale!(X, xmeans, xscales)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
    end
    ## XtY 
    fweightr!(Y, weights.values)
    XtY = X' * Y
    ## Pre-allocation
    T  = similar(X, n, nlv)
    V  = similar(X, p, nlv)
    R  = similar(V)
    C  = similar(X, q, nlv)
    TT = similar(X, nlv)
    t  = similar(X, n)
    dt = similar(t)   
    v  = similar(X, p)
    r  = similar(v)
    c  = similar(X, q)
    zXtY = similar(XtY)
    ## End
    ## de Jong Chemolab 1993 Table 1 (as fast as Appendix) 
    @inbounds for a = 1:nlv
        if a == 1
            zXtY .= XtY
        else
            Pa = vcol(V, 1:(a - 1))
            zXtY .= XtY .- Pa * inv(Pa' * Pa) * Pa' * XtY
        end
        r .= svd!(zXtY).U[:, 1] 
        mul!(t, X, r)                 
        @. dt = weights.values * t            
        tt = dot(t, dt)
        mul!(c, XtY', r)
        c ./= tt                      
        mul!(v, X', dt)
        V[:, a] .= v ./ tt
        T[:, a] .= t
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
    end
    ## B = R * inv(T' * D * T) * T' * D * Y
    ## W does not exist in SIMPLS ==> below it is filled by R (for 'vip')
    Plsr(T, V, R, R, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing, par)
end



