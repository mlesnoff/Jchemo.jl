"""
    outstah(X, P; kwargs...)
    outstah!(X::Matrix, P::Matrix; kwargs...)
Compute the Stahel-Donoho outlierness.
* `X` : X-data (n, p).
* `P` : A projection matrix (p, nlv) representing the directions 
    of the projection pursuit.
Keyword arguments:
* `scal` : Boolean. If `true`, each column of `X` is scaled by MAD 
    before computing the outlierness.

See Maronna and Yohai 1995 for details on the outlierness 
measure. 

A projection-pursuit approach is used: given a projection matrix `P` (p, nlv) 
(in general built randomly), the observations (rows of `X`) are projected on 
the `nlv` directions and the outlierness is computed for each observation 
from these projections.

## References
Maronna, R.A., Yohai, V.J., 1995. The Behavior of the 
Stahel-Donoho Robust Multivariate Estimator. Journal of the 
American Statistical Association 90, 330–341. 
https://doi.org/10.1080/01621459.1995.10476517

## Examples
```julia
n = 300 ; p = 700 ; m = 80
ntot = n + m
X1 = randn(n, p)
X2 = randn(m, p) .+ rand(1:3, p)'
X = vcat(X1, X2)

nlv = 10
P = rand(0:1, p, nlv)
#P = simpphub(X; nsim = 2000)
scal = false
#scal = true
res = outstah(X, P; scal) ;
pnames(res)
res.d  # outlierness 
plotxy(1:nro(X), res.d).f
```
""" 
function outstah(X, P; kwargs...)
    outstah!(copy(ensure_mat(X)), ensure_mat(P); kwargs...)
end

function outstah!(X::Matrix, P::Matrix; kwargs...) 
    par = recovkwargs(Par, kwargs)
    Q = eltype(X)
    n, p = size(X)
    s_scal = ones(Q, p) 
    if par.scal
        s_scal .= colmad(X)
        fscale!(X, s_scal)
    end
    T = X * P  # scaling P by colnorm(P) has no effect on d and T
    #T = X * fscale(P, colnorm(P))
    mu = colmed(T)
    s = colmad(T)
    fcscale!(T, mu, s)
    T .= abs.(T)
    d = similar(T, n)
    @inbounds for i = 1:n
        d[i] = maximum(vrow(T, i))
    end
    (d = d, T, s_scal, mu, s)
end


