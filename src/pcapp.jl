"""
    pcapp(X; kwargs...)
    pcapp!(X::Matrix; kwargs...)
Robust PCA by projection pursuit.
* `X` : X-data (n, p). 
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `nsim` : Nb. of additional (to X-rows) simulated directions for the 
    projection pursuit.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its MAD.

For `nsim = 0`, this is the Croux & Ruiz-Gazen (C-R, 2005) PCA algorithm that 
uses a projection pursuit (PP) method. Data `X` are robustly centered by the 
spatial median, and the observations are projected to specific "PP" directions 
that are defined by the observations (rows of `X`) after they are normed. 
The first PCA loading vector is the direction (within the PP directions) that 
maximizes a given "projection index", here the median absolute deviation (MAD). 
Then, `X` is deflated to this loading vector, and the process is re-run to define
the next loading vector. And so on. 

A possible extension of the algorithm is to randomly simulate additionnal candidate 
PP directions to the n row observations. If `nsim > 0`, the function simulates `nsim` 
additional PP directions to the n initial ones, as proposed in Hubert et al. (2005): 
random couples of observations are sampled in `X` and, for each couple, the direction 
passes through the two observations of the couple (see function `simpphub`).

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
using Jchemo, JchemoData, JLD2, CairoMakie 
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "octane.jld2") 
@load db dat
pnames(dat)
X = dat.X 
wlst = names(X)
wl = parse.(Float64, wlst)
n = nro(X)

nlv = 3
model = pcapp; nlv, nsim = 2000)  
#model = pcasvd; nlv) 
fit!(model, X)
pnames(model)
pnames(model.fm)
@head T = model.fm.T
## Same as:
@head transf(model, X)

i = 1
plotxy(T[:, i], T[:, i + 1]; zeros = true, xlabel = string("PC", i), 
    ylabel = string("PC", i + 1)).f
```
""" 

function pcapp(X; kwargs...)
    pcapp!(copy(ensure_mat(X)); kwargs...)
end

function pcapp!(X::Matrix; kwargs...)
    par = recovkw(ParPcapp, kwargs).par 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    nsim = par.nsim
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
    fobj = colmad
    for a = 1:nlv
        ## For simpphub: the nb. columns of zP can be variable (max = n + A(n, 2))
        zP = fsimpp(X; nsim)  
        zT = X * zP 
        zobj = fobj(zT)
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
    Pca(T, P, sv, xmeans, xscales, weights, nothing, par)
end

