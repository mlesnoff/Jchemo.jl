"""
    sampcla(x, k::Union{Int, Vector{Int}}, 
        y = nothing)
Build training vs. test sets by stratified sampling.  
* `x` : Class membership (n) of the observations.
* `k` : Nb. test observations to sample in each class. 
    If `k` is a single value, the nb. of sampled 
    observations is the same for each class. Alternatively, 
    `k`can be a vector of length equal to the nb. of 
    classes in `x`.
* `y` : Quantitative variable (n) used if systematic sampling.

Two outputs are returned (= row indexes of the data): 
* `train` (n - `k`),
* `test` (`k`). 

If `y` = `nothing`, the sampling is random, else it is 
systematic over the sorted `y`(see function `sampsys`).

## References
Naes, T., 1987. The design of calibration in near infra-red reflectance 
analysis by clustering. Journal of Chemometrics 1, 121-134.

## Examples
```julia
x = string.(repeat(1:3, 5))
n = length(x)
tab(x)
k = 2 
res = sampcla(x, k)
res.test
x[res.test]
tab(x[res.test])

y = rand(n)
res = sampcla(x, k, y)
res.test
x[res.test]
tab(x[res.test])
```
""" 
function sampcla(x, k::Union{Int, Vector{Int}}, 
        y = nothing)
    x = vec(x)
    n = length(x)
    tabx = tab(x)
    lev = tabx.keys
    ni = tabx.vals
    nlev = length(lev)
    length(k) == 1 ? k = repeat([k], nlev) : nothing
    s = list(Vector{Int}, nlev)
    @inbounds for i in 1:nlev
        k[i] = min(k[i], ni[i])
        zs = findall(x .== lev[i])
        if isnothing(y)
            s[i] = sample(zs, k[i]; replace = false)
        else
            y = vec(y)
            u = sampsys(y[zs], k[i]).test
            s[i] = zs[u]
        end
    end
    s = reduce(vcat, s)
    zn = collect(1:n)
    train = zn[setdiff(1:end, s)]
    (train = train, test = s, lev, ni, k)
end


