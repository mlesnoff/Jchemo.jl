"""
    outeucl(X, P; kwargs...)
    outeucl!(X::Matrix, P::Matrix; kwargs...)
Compute an outlierness from Euclidean distances to center.
* `X` : X-data (n, p).
Keyword arguments:
* `scal` : Boolean. If `true`, each column of `X` is scaled by MAD 
    before computing the outlierness.

Outlyingness is calculated by the Euclidean distance between 
the observation (rows of `X`) and a robust estimate of the center of the data 
(the spatial median). 
Such outlyingness was for instance used in the robust PLSR algorithm 
of Serneels et al. 2005 (PRM). 

## References
Serneels, S., Croux, C., Filzmoser, P., Van Espen, P.J., 2005. 
Partial robust M-regression. 
Chemometrics and Intelligent Laboratory Systems 79, 55-64. 
https://doi.org/10.1016/j.chemolab.2005.04.007

## Examples
```julia
n = 300 ; p = 700 ; m = 80
ntot = n + m
X1 = randn(n, p)
X2 = randn(m, p) .+ rand(1:3, p)'
X = vcat(X1, X2)

nlv = 10
scal = false
#scal = true
res = outeucl(X; scal) ;
pnames(res)
res.d    # outlierness 
plotxy(1:ntot, res.dstand).f
```
""" 
function outeucl(X; kwargs...)
    outeucl!(copy(ensure_mat(X)); kwargs...)
end

function outeucl!(X::Matrix; kwargs...) 
    par = recovkwargs(Par, kwargs)
    Q = eltype(X)
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal
        xscales .= colmad(X)
        fscale!(X, xscales)
    end
    xmeans = Jchemo.colmedspa(X)
    d = vec(sqrt.(euclsq(X, xmeans')))
    (d = d, xmeans, xscales)
end


