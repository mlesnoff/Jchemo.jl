"""
    sampks(X, k; metric = :eucl)
Build training/test sets by Kennard-Stone sampling.  
* `X` : X-data (n, p).
* `k` : Nb. observations to sample (= output `test`). 
* `metric` : Metric used for the distance computation.
    Possible values: :eucl, :mah.

Two outputs (= row indexes of the data) are returned: 
* `train` (n - `k`),
* `test` (`k`). 

Output `test` is built from the Kennard-Stone (KS) 
algorithm (Kennard & Stone, 1969). 

**Note:** By construction, the set of observations 
selected by KS sampling contains higher variability than 
the set of the remaining observations. In the seminal 
paper (K&S, 1969), the algorithm is used to select observations
that will be used to build a calibration set. In the present 
function, KS is used to select a test set with higher variability
yhan the training set. 

## References
Kennard, R.W., Stone, L.A., 1969. Computer aided design of experiments. 
Technometrics, 11(1), 137-148.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc

k = 200
res = sampks(X, k)
pnames(res)
res.train 
res.test

fm = pcasvd(X; nlv = 15) ;
res = sampks(fm.T, k; metric = :mah)
```
""" 
function sampks(X, k; metric = :eucl)
    @assert in([:eucl, :mah])(metric) "Wrong value for argument 'metric'."
    k = Int(round(k))
    if metric == :eucl
        D = euclsq(X, X)
    else
        D = mahsq(X, X)
    end
    zn = 1:nro(D)
    ## Initial selection of 2 obs. (train)
    s = findall(D .== maximum(D))
    s = [s[1][1] ; s[1][2]]
    ## Candidates
    can = zn[setdiff(1:end, s)]
    @inbounds for i = 1:(k - 2)
        u = vec(minimum(D[s, can], dims = 1))
        zs = can[findall(u .== maximum(u))[1]]
        s = [s ; zs]
        can = zn[setdiff(1:end, s)]
    end
    sort!(s)
    (train = can, test = s)
end

