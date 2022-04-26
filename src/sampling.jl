"""
    sampks(X; k, metric = "eucl")
Kennard-Stone sampling.  
* `X` : X-data.
* `k` : Nb. observations to sample ==> output `train`. 
* `metric` : Metric used for the distance computation.
    Possible values: "eucl", "mahal".

The function divides the data X in two sets, `train` vs `test`, 
using the Kennard-Stone (KS) algorithm (Kennard & Stone, 1969). 
The two sets have different underlying probability distributions: 
`train` has higher dispersion than `test`.

## References
Kennard, R.W., Stone, L.A., 1969. Computer aided design of experiments. 
Technometrics, 11(1), 137-148.

## Examples
```julia
using JLD2, CairoMakie
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y

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
    if metric == "eucl"
        D = euclsq(X, X)
    elseif metric == "mahal"
        D = mahsq(X, X)
    end
    zn = 1:size(D, 1)
    # Initial 2 selections (train)
    s = findall(D .== maximum(D))
    s = [s[1][1] ; s[1][2]]
    # Candidates
    cand = zn[setdiff(1:end, s)]
    @inbounds for i = 1:(k - 2)
        u = vec(minimum(D[s, cand], dims = 1))
        zs = cand[findall(u .== maximum(u))[1]]
        s = [s ; zs]
        cand = zn[setdiff(1:end, s)]
    end
    (train = s, test = cand)
end
    
"""
    sampdp(X; k, metric = "eucl")
DUPLEX sampling.  
* `X` : X-data (n, p).
* `k` : Nb. pairs of observations to sample. Must be <= n / 2. 
* `metric` : Metric used for the distance computation.
    Possible values are: "eucl", "mahal".

The function divides the data `X` in two sets of equal size, 
`train` vs `test`, using the DUPLEX algorithm (Snee, 1977 p.421).
The two sets are expected to cover approximately the same region and
have similar statistical properties. 

The user may add (a posteriori) the eventual remaining observations 
(output `remain`) to `train`.

## References
Kennard, R.W., Stone, L.A., 1969. Computer aided design of experiments. 
Technometrics, 11(1), 137-148.

Snee, R.D., 1977. Validation of Regression Models: Methods and Examples. 
Technometrics 19, 415-428. https://doi.org/10.1080/00401706.1977.10489581

## Examples
```julia
using JLD2, CairoMakie
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
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
    cand = zn[setdiff(1:end, [s1 ; s2])]
    @inbounds for i = 1:(k - 2)
        u = vec(minimum(D[s1, cand], dims = 1))
        zs = cand[findall(u .== maximum(u))[1]]
        s1 = [s1 ; zs]
        cand = zn[setdiff(1:end, s1)]
        u = vec(minimum(D[s2, cand], dims = 1))
        zs = cand[findall(u .== maximum(u))[1]]
        s2 = [s2 ; zs]
        cand = zn[setdiff(1:end, [s1 ; s2])]
    end
    (train = s1, test = s2, remain = cand)
end
    
"""
    sampsys(y; k)
Systematic sampling over a quantitative variable.  
* `y` : Quantitative variable to sample.
* `k` : Nb. observations to sample ==> output `train`. Must be >= 2.

Systematic sampling (regular grid) over `y`.

The minimum and maximum of `y` are always sampled.

## Examples
```julia
y = rand(7)
[y sort(y)]
res = sampsys(y; k = 3)
y[res.train]
```
""" 
function sampsys(y; k)
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
    test = zn[setdiff(1:end, s)]
    (train = s, test = test)
end

"""
    sampclas(x, y = nothing; k)
Stratified sampling.  
* `x` : Classes of the observations.
* `y` : Quantitative variable used if systematic sampling.
* `k` : Nb. observations to sample in each class ==> output `train`. 

The length of `k` must be either 1 (`k` = equal number of training observations 
to select per class) or the number of classes in `x`.

If `y = nothing` (default), the sampling is random, else it is 
systematic (grid over `y`).

## References
Naes, T., 1987. The design of calibration in near infra-red reflectance analysis by clustering. 
Journal of Chemometrics 1, 121-134.

## Examples
```julia
x = string.(repeat(1:5, 3))
tab(x)
res = sampclas(x; k = 2)
res.train
x[res.train]
tab(x[res.train])

x = string.(repeat(1:5, 3))
n = length(x) ; y = rand(n) 
res = sampclas(x, y; k = 2)
res.train
x[res.train]
tab(x[res.train])
```
""" 
function sampclas(x, y = nothing; k)
    x = vec(x)
    n = length(x)
    res = tab(x)
    lev = res.keys
    nlev = length(lev)
    ni = collect(values(res))
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
    (train = s, test = test, lev = lev, ni = ni, k = k)
end









