"""
    plssimp(X, Y, weights = ones(size(X, 1)); nlv)
Partial Least Squares Regression (PLSR) with the SIMPLS algorithm (de Jong 1993).
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.

`weights` is internally normalized to sum to 1. 

`X` and `Y` are internally centered.

**Note:** In this function, scores T are not normed, conversely the original 
algorithm of de Jong (2013)

## References

de Jong, S., 1993. SIMPLS: An alternative approach to partial least squares 
regression. Chemometrics and Intelligent Laboratory Systems 18, 251–263. 
https://doi.org/10.1016/0169-7439(93)85002-X

""" 
function plssimp(X, Y, weights = ones(size(X, 1)); nlv)
    plssimp!(copy(X), copy(Y), weights; nlv = nlv)
end

function plssimp!(X, Y, weights = ones(size(X, 1)); nlv)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = nco(Y)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
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
    # This is Table 1 (as fast as Appendix) in de Jong 1993
    @inbounds for a = 1:nlv
        if a == 1
            tmp .= XtY
        else
            z = vcol(P, 1:(a - 1))
            tmp .= XtY .- z * inv(z' * z) * z' * XtY
        end
        u = svd!(tmp).U 
        r .= u[:, 1]
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
     Plsr(T, P, R, W, C, TT, xmeans, ymeans, weights, nothing)
end



