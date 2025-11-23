"""
    sampks(X, k::Int; metric = :eucl)
Build training vs. test sets by Kennard-Stone sampling.  
* `X` : X-data (n, p).
* `k` : Nb. test observations to sample.
Keyword arguments: 
* `metric` : Metric used for the distance computation. Possible values are: `:eucl` (Euclidean), 
    `:mah` (Mahalanobis).

Two outputs (= row indexes of the data) are returned: 
* `train` (`n` - `k`),
* `test` (`k`). 

Output `test` is built from the Kennard-Stone (KS) algorithm (Kennard & Stone, 1969). 

**Note:** By construction, the set of observations selected by KS sampling contains higher variability than the set of 
the remaining observations. In the seminal article (K&S, 1969), the algorithm is used to select observations that will 
be used to build a calibration set. To the opposite, in the present function, KS is used to select a test set with 
higher variability than the training set. 

## References
Kennard, R.W., Stone, L.A., 1969. Computer aided design of experiments. Technometrics, 11(1), 137-148.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat

X = dat.X 
y = dat.Y.tbc

k = 80
res = sampks(X, k)
@names res
res.train 
res.test

model = pcasvd(nlv = 15) 
fit!(model, X) 
@head T = model.fitm.T
res = sampks(T, k; metric = :mah)

#####################

n = 10
k = 25 
X = [repeat(1:n, inner = n) repeat(1:n, outer = n)] 
X = Float64.(X) 
X .= X + .1 * randn(nro(X), nco(X))
s = sampks(X, k).test
f, ax = plotxy(X[:, 1], X[:, 2])
scatter!(ax, X[s, 1], X[s, 2]; color = "red") 
f
```
""" 
function sampks(X, k::Int; metric = :eucl)
    @assert in([:eucl, :mah, :sam, :cos, :cor])(metric) "Wrong value for argument 'metric'."
    if metric == :eucl
        D = euclsq(X, X)
    else
        D = mahsqchol(X, X)
    end
    zn = 1:nro(D)
    ## Initial selection of 2 obs. (train)
    s = findall(D .== maximum(D))
    s = [s[1][1] ; s[1][2]]
    ## Candidates
    cand = zn[setdiff(1:end, s)]
    @inbounds for i = 1:(k - 2)
        u = vec(minimum(D[s, cand], dims = 1))
        zs = cand[findall(u .== maximum(u))[1]]
        s = [s ; zs]
        cand = zn[setdiff(1:end, s)]
    end
    sort!(s)
    (train = cand, test = s)
end

