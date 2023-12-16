"""
    sampdp(X, k; metric = :eucl)
Build training/test sets by DUPLEX sampling.  
* `X` : X-data (n, p).
* `k` : Nb. pairs of observations to sample 
    (outputs `train` and `test`). Must be <= n / 2. 
* `metric` : Metric used for the distance computation.
    Possible values are: :eucl, :mah.

Three outputs (= row indexes of the data) are returned: 
* `train` (`k`), 
* `test` (`k`),
* `remain` (n - 2 * `k`). 

Outputs `train` and `test` are built from the DUPLEX algorithm 
(Snee, 1977 p.421). They are expected to cover approximately the same 
X-space region and have similar statistical properties. 

In practice, when output `remain` is not empty (i.e. theer are remaining 
observations), one common strategy is to add it to output `train`.

## References
Kennard, R.W., Stone, L.A., 1969. Computer aided design of experiments. 
Technometrics, 11(1), 137-148.

Snee, R.D., 1977. Validation of Regression Models: Methods and Examples. 
Technometrics 19, 415-428. https://doi.org/10.1080/00401706.1977.10489581

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
n = nro(X)

k = 140
res = sampdp(X, k)
pnames(res)
res.train 
res.test
res.remain

fm = pcasvd(X; nlv = 15)
T = fm.T
res = sampdp(T, k; metric = :mah)

n = 10 ; k = 25 
X = [repeat(1:n, inner = n) repeat(1:n, outer = n)] 
X = Float64.(X) 
X .= X + .1 * randn(nro(X), nco(X))
s = sampks(X, k).train 
f, ax = scatter(X[:, 1], X[:, 2])
scatter!(X[s, 1], X[s, 2], color = "red") 
f
```
""" 
function sampdp(X, k::Int; metric = :eucl)
    @assert in([:eucl, :mah])(metric) "Wrong value for argument 'metric'."
    if metric == :eucl
        D = euclsq(X, X)
    else
        D = mahsqchol(X, X)
    end
    n = size(D, 1)
    zn = 1:n
    ## Initial selection of 2 pairs of obs.
    s = findall(D .== maximum(D)) 
    s1 = [s[1][1] ; s[1][2]]
    zD = copy(D)
    zD[s1, :] .= -Inf ;
    zD[:, s1] .= -Inf ;
    s = findall(D .== maximum(zD)) 
    s2 = [s[1][1] ; s[1][2]]
    ## Candidates
    can = zn[setdiff(1:end, [s1 ; s2])]
    @inbounds for i = 1:(k - 2)
        u = vec(minimum(D[s1, can], dims = 1))
        zs = can[findall(u .== maximum(u))[1]]
        s1 = [s1 ; zs]
        can = zn[setdiff(1:end, s1)]
        u = vec(minimum(D[s2, can], dims = 1))
        zs = can[findall(u .== maximum(u))[1]]
        s2 = [s2 ; zs]
        can = zn[setdiff(1:end, [s1 ; s2])]
    end
    sort!(s1)
    sort!(s2)
    (train = s1, test = s2, remain = can)
end
    
