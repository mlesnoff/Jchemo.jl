"""
    plsrosa(X, Y, weights = ones(nro(X)); nlv,
        scal::Bool = false)
    plsrosa!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        scal::Bool = false)
Partial Least Squares Regression (PLSR) with the ROSA algorithm (Liland et al. 2016).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
* `nlv` : Nb. latent variables (LVs) to consider.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.

**Note:** The function has the following differences with the original 
algorithm of Liland et al. (2016):
* Scores T (LVs) are not normed.
* Multivariate Y is allowed.

See `?plsrosa` for examples.
    
## References
Liland, K.H., Næs, T., Indahl, U.G., 2016. ROSA—a fast extension of partial least 
squares regression for multiblock data analysis. Journal of Chemometrics 30, 
651–662. https://doi.org/10.1002/cem.2824
""" 
function plsrosa(X, Y; par = Par())
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plsrosa(X, Y, weights; par)
end

function plsrosa(X, Y, weights::Weight; par = Par())
    plsrosa!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; par)
end

function plsrosa!(X::Matrix, Y::Matrix, weights::Weight; 
        par = Par())
    n, p = size(X)
    q = nco(Y)
    nlv = min(par.nlv, n, p)
    D = Diagonal(weights.w)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(eltype(X), p)
    yscales = ones(eltype(Y), q)
    if par.scal 
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
    W = similar(X, p, nlv)
    P = copy(W)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t = similar(X, n)
    dt = similar(X, n)   
    zp = similar(X, p)
    w = similar(X, p)
    c = similar(X, q)
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
        if a > 1
            z = vcol(T, 1:(a - 1))
            t .= t .- z * inv(z' * D * z) * z' * (D * t)
            z = vcol(W, 1:(a - 1))
            w = w .- z * (z' * w)
            w ./= sqrt(dot(w, w))
        end
        dt .= weights.w .* t
        tt = dot(t, dt)
        mul!(c, Y', dt)
        c ./= tt                      
        mul!(zp, X', dt)
        zp ./= tt
        Y .-= t * c'
        P[:, a] .= zp  
        T[:, a] .= t
        W[:, a] .= w
        C[:, a] .= c
        TT[a] = tt
     end
     R = W * inv(P' * W)
     Plsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing)
end

