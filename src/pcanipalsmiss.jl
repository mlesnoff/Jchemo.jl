"""
    pcanipalsmiss(X, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    pcanipalsmiss!(X::Matrix, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
PCA by NIPALS algorithm allowing missing data.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. principal components (PCs).
* `gs` : Boolean. If `true` (default), a Gram-Schmidt orthogonalization 
    of the scores and loadings is done. 
* `tol` : Tolerance value for stopping the iterations.
* `maxit` : Maximum nb. iterations.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Let us note D the (n, n) diagonal matrix of `weights` and X the centered 
matrix in metric D. The function minimizes ||X - T * P'||^2  in metric D 
by NIPALS. 

## References
Wright, K., 2018. Package nipals: Principal Components Analysis using NIPALS 
with Gram-Schmidt Orthogonalization. https://cran.r-project.org/

## Examples
```julia
using LinearAlgebra

X = [1. 2 missing 4 ; 4 missing 6 7 ; missing 5 6 13 ; 
    missing 18 7 6 ; 12 missing 28 7] 

tol = 1e-15
nlv = 3 
weights = ones(n) 
#weights = collect(1:n) 
scal = false
#scal = true
gs = false
#gs = true
fm = pcanipalsmiss(X, weights; nlv = nlv, 
    tol = tol, gs = gs, scal = scal, maxit = 500) ;
pnames(fm)
fm.niter
fm.sv
fm.P
fm.T
## Check if orthogonality
fm.P' * fm.P
fm.T' * Diagonal(mweight(weights)) * fm.T

## Impute missing data in x
fm = pcanipalsmiss(X, weights; nlv = 2, 
    gs = true, scal = scal) ;
Xfit = xfit(fm)
s = ismissing.(X)
Xres = copy(X)
Xres[s] .= Xfit[s]
Xres
```
""" 
function pcanipalsmiss(X, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    pcanipalsmiss!(copy(ensure_mat(X)), weights; nlv = nlv, 
        gs = gs, tol = tol, maxit = maxit, scal = scal)
end

function pcanipalsmiss!(X::Matrix, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    n, p = size(X)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    xmeans = colmeanskip(X, weights) 
    #xmeans = colmeanskip(X)
    xscales = ones(p)
    if scal 
        xscales .= colstdskip(X, weights)
        #xscales .= colstdskip(X)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    sqrtw = sqrt.(weights)
    X .= Diagonal(sqrtw) * X
    t = similar(X, n)
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    sv = similar(X, nlv)
    niter = list(nlv, Int64)
    if gs
        UUt = zeros(n, n)
        VVt = zeros(p, p)
    end
    for a = 1:nlv
        if gs == false
            res = nipalsmiss(X; tol = tol, maxit = maxit)
        else
            res = nipalsmiss(X, UUt, VVt; 
                tol = tol, maxit = maxit)
        end
        t .= res.u * res.sv
        T[:, a] .= t ./ sqrtw
        #T[:, a] .= t
        P[:, a] .= res.v           
        sv[a] = res.sv
        niter[a] = res.niter
        X .-= t * res.v'
        if gs
            UUt .+= res.u * res.u' 
            VVt .+= res.v * res.v'
        end
    end    
    Pca(T, P, sv, xmeans, xscales, weights, niter) 
end

