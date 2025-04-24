"""
    outknn(X; metric = :eucl, k, algo = median, scal::Bool = false)
    outknn!(X::Matrix; metric = :eucl, k, algo = median, scal::Bool = false)
Compute a KNN distance-based outlierness.
* `X` : X-data (n, p).
Keyword arguments:
* `metric` : Metric used to compute the dustances. See function `getknn`.
* `k` : Nb. nearest neighbors to consider.
* `algo` : Function summarizing the `k` distances to the neighbours.
* `scal` : Boolean. If `true`, each column of `X` is scaled before computing the outlierness.

For each observation (row of `X`), the outlierness is defined by the summary  (e.g. by median) of the distances between 
the observation and its `k` nearest neighbors.

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

metric = :sam ; k = 20 ; algo = median
#algo = maximum
res = outknn(X, V; scal) ;
@names res
res.d    # outlierness 
plotxy(1:ntot, res.d).f
```
""" 

function outknn(X; metric = :eucl, k, algo = median, scal::Bool = false)
    outknn!(copy(ensure_mat(X)); k, metric, algo, scal)
end

function outknn!(X::Matrix; metric = :eucl, k, algo = median, scal::Bool = false)
    Q = eltype(X)
    n, p = size(X)
    xscales = ones(Q, p)
    if scal
        xscales .= colstd(X)
        fscale!(X, xscales)
    end
    k > n - 1 ? k = n - 1 : nothing
    res = getknn(X, X; k = k + 1, metric)
    d = zeros(n)
    @inbounds for i in eachindex(d)
        d[i] = algo(res.d[i][2:end])
    end
    (d = d, xscales)
end



