"""
    stah(X, a; scal = true)
Stahel-Donoho outlierness.

* `X` : X-data.
* `a` : Nb. dimensions simulated for the projection pursuit method.
* `scal` : Boolean. If `true`, matrix `X` is centred (by median) 
    and scaled (by MAD) before computing the outlierness.

The outlierness measure is computed from a projection-pursuit approach:
* directions in the column-`X` space (linear combinations of the columns 
    of `X`) are randomly simulated, 
* and the observations (rows of `X`) are projected on these directions.

See Maronna and Yohai (1995) for details. 

## References
Maronna, R.A., Yohai, V.J., 1995. The Behavior of the Stahel-Donoho Robust Multivariate Estimator. 
Journal of the American Statistical Association 90, 330–341. https://doi.org/10.1080/01621459.1995.10476517

## Examples
```julia
using StatsBase

n = 300 ; p = 700 ; m = 80 ; ntot = n + m
X1 = randn(n, p)
X2 = randn(m, p) .+ sample(1:3, p)'
X = vcat(X1, X2)

a = 100
res = stah(X, a; scal = true) ;
res.d # outlierness

plotxy(1:nro(X), res.d).f
```
""" 
function stah(X, a; scal = true) 
    zX = copy(ensure_mat(X))
    Q = eltype(zX)
    n, p = size(zX)
    P = reshape(sample(0:1, p * a), p, a)
    mu_scal = zeros(Q, p)
    s_scal = ones(Q, p) 
    if par.scal
        mu_scal .= vec(median(zX, dims = 1))
        s_scal .= colmad(zX)
        cscale!(zX, mu_scal, s_scal)
    end
    T = zX * P
    mu = vec(median(T, dims = 1))
    s = colmad(T)
    cscale!(T, mu, s)
    T .= abs.(T)
    d = similar(T, n)
    @inbounds for i = 1:n
        d[i] = maximum(vrow(T, i))
    end
    (d = d, P, mu_scal, s_scal, mu, s)
end


