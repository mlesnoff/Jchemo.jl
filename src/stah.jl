"""
    stah(X, a; scal = true)
Compute the Stahel-Donoho outlierness.
* `X` : X-data.
* `a` : Nb. dimensions simulated for the 
    projection pursuit method.
Keyword arguments:
* `scal` : Boolean. If `true`, matrix `X` is centred 
    (by median) and scaled (by MAD) before computing 
    the outlierness.

The outlierness measure is computed from a projection-pursuit 
approach:
* directions in the column-`X` space (linear combinations of 
    the columns of `X`) are randomly simulated, 
* and the observations (rows of `X`) are projected on 
    these directions.

See Maronna and Yohai (1995) for details. 

## References
Maronna, R.A., Yohai, V.J., 1995. The Behavior of the Stahel-Donoho Robust 
Multivariate Estimator. Journal of the American Statistical Association 90, 
330–341. https://doi.org/10.1080/01621459.1995.10476517

## Examples
```julia
n = 300 ; p = 700 ; m = 80
ntot = n + m
X1 = randn(n, p)
X2 = randn(m, p) .+ rand(1:3, p)'
X = vcat(X1, X2)

a = 10
scal = false
#scal = true
res = stah(X, a; scal) ;
pnames(res)
res.d
plotxy(1:nro(X), res.d).f

```
""" 
function stah(X, a; scal = true) 
    zX = copy(ensure_mat(X))  # for inplace if scal
    Q = eltype(zX)
    n, p = size(zX)
    P = rand(0:1, p, a)
    mu_scal = zeros(Q, p)
    s_scal = ones(Q, p) 
    if scal
        mu_scal .= vec(median(zX, dims = 1))
        s_scal .= colmad(zX)
        fcscale!(zX, mu_scal, s_scal)
    end
    T = zX * P
    mu = vec(median(T, dims = 1))
    s = colmad(T)
    fcscale!(T, mu, s)
    T .= abs.(T)
    d = similar(T, n)
    @inbounds for i = 1:n
        d[i] = maximum(vrow(T, i))
    end
    (d = d, P, mu_scal, s_scal, mu, s)
end


