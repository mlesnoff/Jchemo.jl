"""
    outstah(X, V; scal = false)
    outstah!(X::Matrix, V::Matrix; scal = false)
Compute the Stahel-Donoho outlierness.
* `X` : X-data (n, p).
* `V` : A projection matrix (p, nlv) representing the directions of the projection pursuit.
Keyword arguments:
* `scal` : Boolean. If `true`, each column of `X` is scaled by its MAD before computing the outlierness.

See Maronna and Yohai 1995 for details on the outlierness measure. 

A projection-pursuit approach is used: given a projection matrix `V` (p, nlv) (in general built randomly), 
the observations (rows of `X`) are projected on the `nlv` directions and the Stahel-Donoho outlierness is computed 
for each observation from these projections.

## References
Maronna, R.A., Yohai, V.J., 1995. The Behavior of the Stahel-Donoho Robust Multivariate 
Estimator. Journal of the American Statistical Association 90, 330–341. 
https://doi.org/10.1080/01621459.1995.10476517

## Examples
```julia
using Jchemo, CairoMakie

n = 300 ; p = 700 ; m = 80
ntot = n + m
X1 = randn(n, p)
X2 = randn(m, p) .+ rand(1:3, p)'
X = vcat(X1, X2)

nlv = 100
V = rand(0:1, p, nlv)
scal = false
#scal = true
res = outstah(X, V; scal) ;
@names res
res.d    # outlierness 
plotxy(1:ntot, res.d).f
```
""" 
function outstah(X, V; scal = false)
    outstah!(copy(ensure_mat(X)), ensure_mat(V); scal)
end

function outstah!(X::Matrix, V::Matrix; scal = false) 
    Q = eltype(X)
    n, p = size(X)
    xscales = ones(Q, p) 
    if scal
        xscales .= colmad(X)
        fscale!(X, xscales)
    end
    ## Scaling V by colnorm(V) has no effect on d and T
    #T = X * fscale(V, colnorm(V))
    T = X * V  
    mu = colmed(T)
    s = colmad(T)  
    fcscale!(T, mu, s)
    T[isnan.(T)] .= convert(Q, 0)
    T .= abs.(T)
    d = similar(T, n)
    @inbounds for i in axes(X, 1)
        d[i] = maximum(vrow(T, i))
    end
    (d = d, T, mu, s, xscales)
end


