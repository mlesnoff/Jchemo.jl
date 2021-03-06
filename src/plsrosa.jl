"""
    plsrosa(X, Y, weights = ones(size(X, 1)); nlv)
    plsrosa!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); nlv)
Partial Least Squares Regression (PLSR) with the ROSA algorithm (Liland et al. 2016).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations.
* `nlv` : Nb. latent variables (LVs) to consider.

`weights` is internally normalized to sum to 1. 
    
`X` and `Y` are internally centered. 

**Note:** The function has the following differences with the original 
algorithm of Liland et al. (2016):
* Scores T are not normed.
* Multivariate Y is allowed.

See `?plskern` for examples.
    
## References
Liland, K.H., Næs, T., Indahl, U.G., 2016. ROSA—a fast extension of partial least 
squares regression for multiblock data analysis. Journal of Chemometrics 30, 
651–662. https://doi.org/10.1002/cem.2824
""" 
function plsrosa(X, Y, weights = ones(size(X, 1)); nlv)
    plsrosa!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv)
end

function plsrosa!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); nlv)
    n, p = size(X)
    q = nco(Y)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    D = Diagonal(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    # Pre-allocation
    XtY = similar(X, p, q)
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    W = copy(P)
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
            w ./= sqrt(dot(w, w))
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
        dt .= weights .* t
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
     Plsr(T, P, R, W, C, TT, xmeans, ymeans, weights, nothing)
end

