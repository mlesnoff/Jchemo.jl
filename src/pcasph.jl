"""
    pcasph(X, weights = ones(nro(X)); nlv, typc = "medspa", 
        delta = .001, scal::Bool = false)
    pcasph!(X, weights = ones(nro(X)); nlv, typc = "medspa", 
        delta = .001, scal::Bool = false)
Spherical PCA.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. principal components (PCs).
* `typc` : Type of centering.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Spherical PCA (Locantore et al. 1990, Maronna 2005, Daszykowski et al. 2007). 
The spatial median used for centering matrix \eqn{X} is calculated by function
`Jchemo.colmedspa`.

## References
Daszykowski, M., Kaczmarek, K., Vander Heyden, Y., Walczak, B., 2007. 
Robust statistics in data analysis - A review. Chemometrics and Intelligent 
Laboratory Systems 85, 203-219. https://doi.org/10.1016/j.chemolab.2006.06.016

Locantore N., Marron J.S., Simpson D.G., Tripoli N., Zhang J.T., Cohen K.L.
Robust principal component analysis for functional data, Test 8 (1999) 1–7

Maronna, R., 2005. Principal components and orthogonal regression based on 
robust scales, Technometrics, 47:3, 264-273, DOI: 10.1198/004017005000000166

## Examples
```julia
using JLD2, JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/octane.jld2") 
@load db dat
pnames(dat)
  
X = dat.X 
wl_str = names(X)
wl = parse.(Float64, wl_str)
n = nro(X)

nlv = 6
fm = pcasph(X; nlv = nlv) ; 
#fm = pcasvd(X; nlv = nlv) ; 
pnames(fm)
T = fm.T

i = 1
plotxy(T[:, i], T[:, i + 1]); zeros = true,
    xlabel = "PC1", ylabel = "PC2").f
```
""" 
function pcasph(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    pcasph(X, weights; values(kwargs)...)
end

function pcasph(X, weights::Weight; kwargs...)
    pcasph!(copy(ensure_mat(X)), weights; values(kwargs)...)
end

function pcasph!(X::Matrix, weights::Weight; 
        par = Par())
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = colmedspa(X; 
        delta = convert(Q, 1e-6))
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    sqrtw = sqrt.(weights.w)
    tX = Matrix(X')
    xnorms = colnorm(tX)
    scale!(tX, xnorms)
    zX = tX'
    res = LinearAlgebra.svd!(sqrtw .* zX)
    P = res.V[:, 1:nlv]
    T = zX * P
    sv = colmad(T)
    #zT = zX * P
    #sv = colmad(zT)
    T .= X * P
    s = sortperm(sv; rev = true)
    T .= T[:, s]
    P .= P[:, s]
    sv .= sv[s]
    Pca(T, P, sv, xmeans, xscales, weights, nothing) 
end
