"""
    sampcla(x, k, y = nothing)
Split training/test sets by stratified sampling.  
* `x` : Class membership (n) of the observations.
* `k` : Nb. observations to sample in each class (output `train`). 
    If `k` is a single value, the nb. sampled observations is the same 
    for each class. Alternatively, `k` can be a vector of length 
    equal to the nb. classes in `x`.
* `y` : Quantitative variable (n) used if systematic sampling.

Two outputs (indexes) are returned: 
* `train` (`k`),
* `test` (n - `k`). 

If `y = nothing` (default), the sampling is random, else it is 
systematic (grid over `y`).

## References
Naes, T., 1987. The design of calibration in near infra-red reflectance analysis by clustering. 
Journal of Chemometrics 1, 121-134.

## Examples
```julia
x = string.(repeat(1:5, 3))
tab(x)
res = sampcla(x, 2)
res.train
x[res.train]
tab(x[res.train])
tab(x[res.test])

x = string.(repeat(1:5, 3))
n = length(x) ; y = rand(n) 
[x y]
res = sampcla(x, 2, y)
res.train
x[res.train]
tab(x[res.train])
```
""" 
function sampcla(x, k, y = nothing)
    k = Int(round(k))
    x = vec(x)
    n = length(x)
    ztab = tab(x)
    lev = ztab.keys
    ni = ztab.vals
    nlev = length(lev)
    isequal(length(k), 1) ? k = fill(k[1], nlev) : nothing
    s = list(nlev, Vector{Int})
    @inbounds for i in 1:nlev
        k[i] = min(k[i], ni[i])
        zs = findall(x .== lev[i])
        if isnothing(y)
            s[i] = sample(zs, k[i]; replace = false)
        else
            y = vec(y)
            u = sampsys(y[zs], k[i]).train
            s[i] = zs[u]
        end
    end
    s = reduce(vcat, s)
    zn = collect(1:n)
    test = zn[setdiff(1:end, s)]
    (train = s, test, lev, ni, k)
end


