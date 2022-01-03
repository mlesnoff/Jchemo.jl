"""
    plsnipals(X, Y, weights = ones(size(X, 1)); nlv)
Partial Least Squares Regression (PLSR) with the NIPALS algorithm 
(e.g. Tenenhaus 1998, Wold 2002).
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to consider.

In the function, the usual PLS2 NIPALS iterations are replaced by a 
direct computation of the PLS weights (`w`) by SVD decomposition of matrix X'Y 
(Hoskuldsson 1988 p.213).

Vector `weights` (row-weighting) is internally normalized to sum to 1. 
See the help of `plskern` for details.

`X` and `Y` are internally centered. 

## References

Hoskuldsson, A., 1988. PLS regression methods. Journal of Chemometrics 2, 211-228.
https://doi.org/10.1002/cem.1180020306

Tenenhaus, M., 1998. La régression PLS: thÃ©orie et pratique. Editions Technip, 
Paris, France.

Wold, S., Sjostrom, M., Eriksson, l., 2001. PLS-regression: a basic tool 
for chemometrics. Chem. Int. Lab. Syst., 58, 109-130.
""" 
function plsnipals(X, Y, weights = ones(size(X, 1)); nlv)
    plsnipals!(copy(X), copy(Y), weights; nlv = nlv)
end

function plsnipals!(X, Y, weights = ones(size(X, 1)); nlv)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)
    nlv = min(nlv, n, p)
    weights = mweights(weights)
    D = Diagonal(weights)
    xmeans = colmeans(X, weights) 
    ymeans = colmeans(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    # Pre-allocation
    XtY = similar(X, p, q)
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    W = copy(P)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = similar(X, n)   
    zp  = similar(X, p)
    w   = similar(X, p)
    c   = similar(X, q)
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
        dt .= weights .* t
        tt = dot(t, dt)
        mul!(c, Y', dt)
        c ./= tt                      
        mul!(zp, X', dt)
        zp ./= tt
        X .-= t * zp'
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

