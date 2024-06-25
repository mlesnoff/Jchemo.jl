"""
    pcaout(X; kwargs...)
    pcaout(X, weights::Weight; kwargs...)
    pcaout!(X::Matrix, weights::Weight; kwargs...)
Robust PCA using outlierness.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.


## References

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
mod = model(pcaout; nlv)  
#mod = model(pcasvd; nlv) 
fit!(mod, X)
pnames(mod)
pnames(mod.fm)
@head T = mod.fm.T
## Same as:
transf(mod, X)

i = 1
plotxy(T[:, i], T[:, i + 1]; zeros = true, xlabel = string("PC", i), 
    ylabel = string("PC", i + 1)).f
```
""" 
function pcaout(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    pcaout(X, weights; kwargs...)
end

function pcaout(X, weights::Weight; kwargs...)
    pcaout!(copy(ensure_mat(X)), weights; kwargs...)
end

function pcaout!(X::Matrix, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs) 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = colmedspa(X)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colmad(X)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    prm = .20
    nlvout = 30
    P = rand(0:1, p, nlvout)
    res = outstah(X, P; scal = true)
    w = talworth(res.d; a = quantile(res.d, 1 - prm))
    res = outeucl(X; scal = true)
    w .*= talworth(res.d; a = quantile(res.d, 1 - prm))
    w .*= weights.w
    w[isequal.(w, 0)] .= 1e-10
    pcasvd(X, mweight(w); nlv)
    ## Final step
    #res = occsdod(fm, X)            
    #v = talworth(res.d.dstand; a = 1)
    #w .*= v
    #w[isequal.(w, 0)] .= 1e-10
    #pcasvd(X, mweight(weights.w .* w); nlv)
end
