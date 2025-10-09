"""
    pcasph(; kwargs...)
    pcasph(X; kwargs...)
    pcasph(X, weights::Weight; kwargs...)
    pcasph!(X::Matrix, weights::Weight; kwargs...)
Spherical PCA.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

Spherical PCA (Locantore et al. 1990, Maronna 2005, Daszykowski et al. 2007). Matrix `X` is centered by the spatial 
median computed by function`Jchemo.colmedspa`.

## References
Daszykowski, M., Kaczmarek, K., Vander Heyden, Y., Walczak, B., 2007. Robust statistics in data analysis - A review. 
Chemometrics and Intelligent Laboratory Systems 85, 203-219. https://doi.org/10.1016/j.chemolab.2006.06.016

Locantore N., Marron J.S., Simpson D.G., Tripoli N., Zhang J.T., Cohen K.L. Robust principal component analysis for 
functional data, Test 8 (1999) 1–7

Maronna, R., 2005. Principal components and orthogonal regression based on robust scales, Technometrics, 47:3, 264-273,
DOI: 10.1198/004017005000000166

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie 
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "octane.jld2") 
@load db dat
@names dat
X = dat.X 
wlst = names(X)
wl = parse.(Float64, wlst)
n = nro(X)

nlv = 3
model = pcasph(; nlv)  
#model = pcasvd(; nlv) 
fit!(model, X)
@names model
@names model.fitm
@head T = model.fitm.T
## Same as:
transf(model, X)

i = 1
plotxy(T[:, i], T[:, i + 1]; zeros = true, xlabel = string("PC", i), ylabel = string("PC", i + 1)).f
```
""" 
pcasph(; kwargs...) = JchemoModel(pcasph, nothing, kwargs)

function pcasph(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    pcasph(X, weights; kwargs...)
end

function pcasph(X, weights::Weight; kwargs...)
    pcasph!(copy(ensure_mat(X)), weights; kwargs...)
end

function pcasph!(X::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPca, kwargs).par 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = Jchemo.colmedspa(X, delta = 0.001)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colmad(X)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    ## Sphere
    Xtt = fscale(X', rownorm(X))'    # X-data projected on the sphere (each row has norm = 1)
    rweight!(Xtt, sqrt.(weights.w))
    res = LinearAlgebra.svd!(Matrix(Xtt)) 
    V = res.V[:, 1:nlv]
    ## End
    T = X * V      
    sv = colmad(T)  # Maronna 2005 p.268 eq.20 [different than in rnirs/rchemo ==> to fix in rchemo]
    s = sortperm(sv; rev = true)
    T .= T[:, s]
    V .= V[:, s]
    sv .= sv[s]
    Pca(T, V, sv, xmeans, xscales, weights, nothing, par) 
end

