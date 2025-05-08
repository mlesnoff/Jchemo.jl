"""
    outeucl(X; scal = false)
    outeucl!(X::Matrix; scal = false)
Compute outlierness from Euclidean distances to center.
* `X` : X-data (n, p).
Keyword arguments:
* `scal` : Boolean. If `true`, each column of `X` is scaled by its MAD before computing the outlierness.

Outlyingness is calculated by the Euclidean distance between the observation (rows of `X`) and a robust estimate 
of the center of the data (in the present function, the spatial median). Such outlyingness was for instance used in the robust 
PLSR algorithm of Serneels et al. 2005 (PRM). 

## References
Serneels, S., Croux, C., Filzmoser, V., Van Espen, V.J., 2005. Partial robust M-regression. 
Chemometrics and Intelligent Laboratory Systems 79, 55-64. 
https://doi.org/10.1016/j.chemolab.2005.04.007

## Examples
```julia
using Jchemo, CairoMakie
n = 300 ; p = 700 ; m = 80
ntot = n + m
X1 = randn(n, p)
X2 = randn(m, p) .+ rand(1:3, p)'
X = vcat(X1, X2)

scal = false
#scal = true
res = outeucl(X; scal) ;
@names res
res.d    # outlierness 
plotxy(1:ntot, res.d).f
```
""" 
function outeucl(X; scal = false)
    outeucl!(copy(ensure_mat(X)); scal)
end

function outeucl!(X::Matrix; scal = false) 
    Q = eltype(X)
    p = nco(X)
    xscales = ones(Q, p)
    if scal
        xscales .= colmad(X)
        fscale!(X, xscales)
    end
    xmeans = Jchemo.colmedspa(X)
    d = vec(sqrt.(euclsq(X, xmeans')))
    (d = d, xmeans, xscales)
end


