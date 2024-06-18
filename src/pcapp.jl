"""
    pcapp(X; kwargs...)
    pcapp!(X::Matrix; kwargs...)
Robust PCA by projection pursuit.
* `X` : X-data (n, p). 
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `nsim` : Nb. of additional simulated directions (to X-rows) for the 
    projection pursuit.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Croux & Ruiz-Gazen (C-R) PCA algorithm using projection pursuit (PP) method 
(Croux & Ruiz-Gazen 2005). The observations are robustly centered, and projected 
to specific "PP" directions (see below) of the space spaned by the variables (X columns). 
The PCA loading vectors are the directions (within the PP directions) that maximize a 
given "projection index", usually a robust spread measure such as MAD. The 1st loading 
vector is choosen within the n directions corresponding the observations (rows of 
X). The next loading vector is choosen in n directions corresponding the rows of the deflated matrix 
X). And so on.

A possible extension of the algorithm is to randomly simulate additionnal candidate 
PP directions to the 
n row observations. In function pca_cr, this is done when argument nsim > 0. In such a case, 
the function simulates nsim additional PP directions to the n initial ones, as proposed in 
Hubert et al. (2005): random couples of observations are sampled in X and, for each couple, 
the direction passes through the two observations of the couple (see functions .simpp.hub 
in file zfunctions.R).

## References
Croux, C., Ruiz-Gazen, A., 2005. High breakdown estimators for 
principal components: the projection-pursuit approach revisited. 
Journal of Multivariate Analysis 95, 206–226. 
https://doi.org/10.1016/j.jmva.2004.08.002

Hubert, M., Rousseeuw, P.J., Vanden Branden, K., 2005. ROBPCA: 
A New Approach to Robust Principal Component Analysis. 
Technometrics 47, 64-79. https://doi.org/10.1198/004017004000000563

## Examples
```julia
using JchemoData, JLD2, CairoMakie 
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "octane.jld2") 
@load db dat
pnames(dat)
X = dat.X 
wlst = names(X)
wl = parse.(Float64, wlst)
n = nro(X)

nlv = 6
mod = model(pcapp; nlv)  
#mod = model(pcasvd; nlv) 
fit!(mod, X)
pnames(mod)
pnames(mod.fm)
@head T = mod.fm.T
## Same as:
transf(mod, X)

i = 1
plotxy(T[:, i], T[:, i + 1]; zeros = true, xlabel = "PC1", 
    ylabel = "PC2").f
```
""" 

function pcapp(X; nsim = 2000, kwargs...)
    pcapp!(copy(ensure_mat(X)); nsim, kwargs...)
end

function pcapp!(X::Matrix; nsim = 2000, kwargs...)
    par = recovkwargs(Par, kwargs) 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = Jchemo.colmedspa(X) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colmad(X)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    t = similar(X, n)
    zp = similar(X, p)
    sv = similar(X, nlv)
    fsimpp = simpphub
    #fsimpp = simppsph   # ~ same for large nsim (~ >= 2000)
    for a = 1:nlv
        ## For simpphub: the nb. columns of zP can be variable (max = n + A(n, 2))
        zP = fsimpp(X; nsim)  
        zT = X * zP 
        zobj = colmad(zT)
        zobj[isnan.(zobj)] .= 0
        s = findall(zobj .== maximum(zobj))[1]
        sv[a] = zobj[s]
        t .= vcol(zT, s)
        zp .= vcol(zP, s)
        T[:, a] = t
        P[:, a] = zp
        X .-= t * zp'
    end
    s = sortperm(sv; rev = true)
    T .= T[:, s]
    P .= P[:, s]
    sv .= sv[s]
    weights = mweight(ones(n))
    Pca(T, P, sv, xmeans, xscales, weights, nothing, kwargs, par)
end

