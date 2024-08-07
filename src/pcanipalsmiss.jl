"""
    pcanipals(X; kwargs...)
    pcanipals(X, weights::Weight; kwargs...)
    pcanipals!(X::Matrix, weights::Weight; kwargs...)
PCA by NIPALS algorithm allowing missing data.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `gs` : Boolean. If `true` (default), a Gram-Schmidt 
    orthogonalization of the scores and loadings is done
    before each X-deflation. 
* `tol` : Tolerance value for stopping 
    the iterations.
* `maxit` : Maximum nb. of iterations.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

## References
Wright, K., 2018. Package nipals: Principal Components Analysis 
using NIPALS with Gram-Schmidt Orthogonalization. 
https://cran.r-project.org/

## Examples
```julia
X = [1 2. missing 4 ; 4 missing 6 7 ; 
    missing 5 6 13 ; missing 18 7 6 ; 
    12 missing 28 7] 

nlv = 3 
tol = 1e-15
scal = false
#scal = true
gs = false
#gs = true
mod = model(pcanipalsmiss; nlv, tol, gs, maxit = 500, scal)
fit!(mod, X)
pnames(mod) 
pnames(mod.fm)
fm = mod.fm ;
fm.niter
fm.sv
fm.P
fm.T
## Orthogonality 
## only if gs = true
fm.T' * fm.T
fm.P' * fm.P

## Impute missing data in X
mod = model(pcanipalsmiss; nlv = 2, gs = true) ;
fit!(mod, X)
Xfit = xfit(mod.fm)
s = ismissing.(X)
X_imput = copy(X)
X_imput[s] .= Xfit[s]
X_imput
```
""" 
function pcanipalsmiss(X; kwargs...)
    z = vec(Matrix(X))
    s = ismissing.(z) .== 0
    Q = eltype(z[s][1, 1])
    weights = mweight(ones(Q, nro(X)))
    pcanipalsmiss(X, weights; kwargs...)
end

function pcanipalsmiss(X, weights::Weight; kwargs...)
    pcanipalsmiss!(copy(ensure_mat(X)), weights; kwargs...)
end

function pcanipalsmiss!(X::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPcanipals, kwargs).par 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = colmeanskip(X, weights) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstdskip(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    sqrtw = sqrt.(weights.w)
    X .= Diagonal(sqrtw) * X
    t = similar(X, n)
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    sv = similar(X, nlv)
    niter = list(Int, nlv)
    if par.gs
        UUt = zeros(n, n)
        VVt = zeros(p, p)
    end
    for a = 1:nlv
        if par.gs == false
            res = nipalsmiss(X; kwargs...)
        else
            res = nipalsmiss(X, UUt, VVt; 
                kwargs...)
        end
        t .= res.u * res.sv
        T[:, a] .= t ./ sqrtw
        #T[:, a] .= t
        P[:, a] .= res.v           
        sv[a] = res.sv
        niter[a] = res.niter
        X .-= t * res.v'
        if par.gs
            UUt .+= res.u * res.u' 
            VVt .+= res.v * res.v'
        end
    end
    Pca(T, P, sv, xmeans, xscales, weights, niter, par) 
end

