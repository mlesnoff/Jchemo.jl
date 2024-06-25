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
    n, p = size(X)
    nlvout = 30
    P = rand(0:1, p, nlvout)
    d = similar(X, n)
    d .= outstah(X, P; scal = par.scal).d
    w = talworth(d; a = quantile(d, 1 - par.prm))
    d .= outeucl(X; scal = par.scal).d
    w .*= talworth(d; a = quantile(d, 1 - par.prm))
    w .*= weights.w
    w[isequal.(w, 0)] .= 1e-10
    pcasvd(X, mweight(w); kwargs...)
end
