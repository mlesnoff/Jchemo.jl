"""
    plsnipals(; kwargs...)
    plsnipals(X, Y; kwargs...)
    plsnipals(X, Y, weights::ProbabilityWeights; kwargs...)
    plsnipals!(X::Matrix, Y::Matrix, weights::ProbabilityWeights; kwargs...)
Partial Least Squares Regression (PLSR) with the Nipals algorithm.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation.

In this function, for PLS2 (multivariate Y), the Nipals iterations are replaced by a direct computation of the 
PLS weights (w) by SVD decomposition of matrix X'Y (Hoskuldsson 1988 p.213).

See function `plskern` for examples.

## References
Hoskuldsson, A., 1988. PLS regression methods. Journal of Chemometrics 2, 211-228.
https://doi.org/10.1002/cem.1180020306

Tenenhaus, M., 1998. La régression PLS: thÃ©orie et pratique. Editions Technip, Paris, France.

Wold, S., Sjostrom, M., Eriksson, l., 2001. PLS-regression: a basic tool for chemometrics. 
Chem. Int. Lab. Syst., 58, 109-130.
""" 
plsnipals(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)

function plsnipals(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = pweight(ones(Q, n))
    plsnipals(X, Y, weights; kwargs...)
end

function plsnipals(X, Y, weights::ProbabilityWeights; kwargs...)
    plsnipals!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plsnipals!(X::Matrix, Y::Matrix, weights::ProbabilityWeights; kwargs...)
    par = recovkw(ParPlsr, kwargs).par
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
    ## Pre-allocation
    XtY = similar(X, p, q)
    T = similar(X, n, nlv)
    W = similar(X, p, nlv)
    V = copy(W)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = similar(X, n)   
    w = similar(X, p)
    v  = copy(w)
    c   = similar(X, q)
    ## End
    @inbounds for a = 1:nlv
        XtY .= X' * fweightr(Y, weights.values)
        if q == 1
            w .= vec(XtY)
            w ./= normv(w)
        else
            w .= svd!(XtY).U[:, 1]
        end
        mul!(t, X, w)
        dt .= weights.values .* t
        tt = dot(t, dt)
        mul!(v, X', dt)
        v ./= tt
        mul!(c, Y', dt)
        c ./= tt                      
        ## Deflation with respect to t: asymetric PLS
        X .-= t * v'
        Y .-= t * c'
        ## End
        V[:, a] .= v  
        T[:, a] .= t
        W[:, a] .= w
        C[:, a] .= c
        TT[a] = tt
    end
    R = W * inv(V' * W)
    Plsr(T, V, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing, par)
end

