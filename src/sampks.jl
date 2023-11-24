"""
    sampks(X, k; metric = :eucl)
Split training/test sets by Kennard-Stone sampling.  
* `X` : X-data (n, p).
* `k` : Nb. observations to sample (output `train`). 
* `metric` : Metric used for the distance computation.
    Possible values: :eucl, :mah.

Two outputs (indexes) are returned: 
* `train` (`k`),
* `test` (n - `k`). 

Output `train` is built using the Kennard-Stone (KS) algorithm (Kennard & Stone, 1969). 
After KS, the two outputs have different underlying probability distributions: `train` has higher 
dispersion than `test`.

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
    k = Int(round(k))
    if metric == :eucl
        D = euclsq(X, X)
    elseif metric == :mah
        D = mahsq(X, X)
    end
    zn = 1:nro(D)
    # Initial 2 selections (train)
    s = findall(D .== maximum(D))
    s = [s[1][1] ; s[1][2]]
    # Candidates
    can = zn[setdiff(1:end, s)]
    @inbounds for i = 1:(k - 2)
        u = vec(minimum(D[s, can], dims = 1))
        zs = can[findall(u .== maximum(u))[1]]
        s = [s ; zs]
        can = zn[setdiff(1:end, s)]
    end
    sort!(s)
    (train = s, test = can)
end

