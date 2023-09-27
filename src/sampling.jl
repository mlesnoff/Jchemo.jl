"""
    sampks(X; k, metric = "eucl")
Kennard-Stone sampling.  
* `X` : X-data (n, p).
* `k` : Nb. observations to sample (output `train`). 
* `metric` : Metric used for the distance computation.
    Possible values: "eucl", "mahal".

Two outputs (indexes) are returned: `train` (`k`) and `test` (n - `k`). 
Output `train` is built using the Kennard-Stone (KS) algorithm (Kennard & Stone, 1969). 
The two sets have different underlying probability distributions: `train` has higher 
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
res = sampks(X; k = k)
pnames(res)
res.train 
res.test

fm = pcasvd(X; nlv = 15)
T = fm.T
res = sampks(T; k = k, metric = "mahal")
```
""" 
function sampks(X; k, metric = "eucl")
    k = Int64(round(k))
    if metric == "eucl"
        D = euclsq(X, X)
    elseif metric == "mahal"
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
    
"""
    sampdp(X; k, metric = "eucl")
DUPLEX sampling.  
* `X` : X-data (n, p).
* `k` : Nb. pairs of observations to sample. Must be <= n / 2. 
* `metric` : Metric used for the distance computation.
    Possible values are: "eucl", "mahal".

Three outputs (indexes) are returned: 
`train` (`k`), `test` (`k`) and `remain` (n - 2 * `k`). 

Outputs `train` and `test` and are built using the DUPLEX algorithm 
(Snee, 1977 p.421). They have equal size and are expected to cover 
approximately the same X-space region and have similar statistical properties. 

In practice, when output `remain` is not empty (remaining observations), 
one strategy is to add it to output `train`.

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
res = sampdp(X; k = k)
pnames(res)
res.train 
res.test
res.remain

fm = pcasvd(X; nlv = 15)
T = fm.T
res = sampdp(T; k = k, metric = "mahal")

n = 10 ; k = 25 
X = [repeat(1:n, inner = n) repeat(1:n, outer = n)] 
X = Float64.(X) 
X .= X + .1 * randn(nro(X), nco(X))
s = sampks(X; k = k).train 
f, ax = scatter(X[:, 1], X[:, 2])
scatter!(X[s, 1], X[s, 2], color = "red") 
f
```
""" 
function sampdp(X; k, metric = "eucl")
    k = Int64(round(k))
    if(metric == "eucl")
        D = euclsq(X, X)
    elseif(metric == "mahal")
        D = mahsq(X, X)
    end
    n = size(D, 1)
    zn = 1:n
    # Initial selection of 2 pairs
    s = findall(D .== maximum(D)) 
    s1 = [s[1][1] ; s[1][2]]
    zD = copy(D)
    zD[s1, :] .= -Inf ;
    zD[:, s1] .= -Inf ;
    s = findall(D .== maximum(zD)) 
    s2 = [s[1][1] ; s[1][2]]
    # Candidates
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
    
"""
    samprand(y; k)
Random sampling.  
* `n` : Total nb. observations.
* `k` : Nb. observations to sample (output `train`).

Two outputs (indexes) are returned: `train` (`k`) and `test` (`n - k`). 
Output `train` is built by random sampling within `1:n`, with no replacement. 

## Examples
```julia
n = 10
samprand(n; k = 3)
```
""" 
function samprand(n; k)
    k = Int64(round(k))
    zn = collect(1:n)
    s = sample(zn, k; replace = false)
    sort!(s)
    test = zn[setdiff(1:end, s)]
    (train = s, test = test)
end

"""
    sampsys(y; k)
Systematic sampling over a quantitative variable.  
* `y` : Quantitative variable (n) to sample.
* `k` : Nb. observations to sample (output `train`). 
    Must be >= 2.

Two outputs (indexes) are returned: `train` (`k`) and `test` (n - `k`). 
Output `train` is built by systematic sampling (regular grid) over `y`. 
It always contains the indexes of the minimum and maximum of `y`.

## Examples
```julia
y = rand(7)
sort(y)
res = sampsys(y; k = 3)
sort(y[res.train])
```
""" 
function sampsys(y; k)
    k = Int64(round(k))
    y = vec(y)
    n = length(y)
    nint = k - 1            # nb. intervals
    alpha = (n - 1) / nint  # step
    z = collect(1:alpha:n)
    z = Int64.(round.(z))
    z = unique(z)
    id = sortperm(y)
    zn = collect(1:n)
    s = zn[id[z]]
    sort!(s)
    test = zn[setdiff(1:end, s)]
    (train = s, test = test)
end

"""
    sampcla(x, y = nothing; k)
Stratified sampling.  
* `x` : Class membership (n) of the observations.
* `y` : Quantitative variable (n) used if systematic sampling.
* `k` : Nb. observations to sample in each class (output `train`). 
    If `k` is a single value, the nb. sampled observations is the same 
    for each class. Alternatively, `k` can be a vector of length 
    equal to the nb. classes in `x`.

If `y = nothing` (default), the sampling is random, else it is 
systematic (grid over `y`).

## References
Naes, T., 1987. The design of calibration in near infra-red reflectance analysis by clustering. 
Journal of Chemometrics 1, 121-134.

## Examples
```julia
x = string.(repeat(1:5, 3))
tab(x)
res = sampcla(x; k = 2)
res.train
x[res.train]
tab(x[res.train])

x = string.(repeat(1:5, 3))
n = length(x) ; y = rand(n) 
res = sampcla(x, y; k = 2)
res.train
x[res.train]
tab(x[res.train])
```
""" 
function sampcla(x, y = nothing; k)
    k = Int64(round(k))
    x = vec(x)
    n = length(x)
    ztab = tab(x)
    lev = ztab.keys
    ni = ztab.vals
    nlev = length(lev)
    isequal(length(k), 1) ? k = fill(k[1], nlev) : nothing
    s = list(nlev, Vector{Int64})
    @inbounds for i in 1:nlev
        k[i] = min(k[i], ni[i])
        zs = findall(x .== lev[i])
        if isnothing(y)
            s[i] = sample(zs, k[i]; replace = false)
        else
            y = vec(y)
            u = sampsys(y[zs]; k = k[i]).train
            s[i] = zs[u]
        end
    end
    s = reduce(vcat, s)
    zn = collect(1:n)
    test = zn[setdiff(1:end, s)]
    (train = s, test, lev, ni, k)
end


