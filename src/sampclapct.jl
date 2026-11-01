"""
    sampclapct(x::Vector{String}, pct::Union{Q, Vector{Q}}, y = nothing; 
        seed::Union{Nothing, Int} = nothing) where Q <: Real
Build training vs. test sets using a stratified (proportion) sampling.  
* `x` : Class membership (n) of the observations. Must be a `Vector{String}`.
* `pct` : Proportion of test observations to sample in each class. If `pct` is a single value, the proportion
    of sampled observations is the same for each class. Alternatively, `pct` can be a vector of length equal 
    to the nb. of classes in `x`.
* `y` : Quantitative variable (n) used if systematic sampling.
Keyword arguments:
* `seed` : Eventual seed for the `Random.MersenneTwister` generator.  

In each class 'i' (with 'ni' observations), the function samples ki = `round(Int, pct * ni)`, or 
ki = `round(Int, pct[i] * ni)` if `pct` is a vector, test observations.

Two outputs are returned (= row indexes of the data): 
* `train` (n - m),
* `test` (m), 
where m = sum(ki).

If `y = nothing`, the sampling of the test observations in each class is random, else it is 
systematic over the sorted `y` (see the principle in function `sampsys`).

# References
Naes, T., 1987. The design of calibration in near infra-red reflectance analysis by clustering. 
Journal of Chemometrics 1, 121-134.

# Examples
```julia
using Jchemo
x = string.(repeat(1:3, 5))
n = length(x)
tab(x)
pct = 0.4
res = sampclapct(x, pct)
res.test
x[res.test]
tab(x[res.test])

sampclapct(x, pct; seed = 123)

y = rand(n)
res = sampclapct(x, pct, y)
res.test
x[res.test]
tab(x[res.test])
```
""" 
function sampclapct(x::Vector{String}, pct::Union{Q, Vector{Q}}, y = nothing; 
        seed::Union{Nothing, Int} = nothing) where Q <: Real
    tabx = tab(x)
    lev = tabx.lev
    ni = tabx.n
    nlev = length(lev)
    if length(pct) == 1
        pct = fill(pct, nlev)
    end
    k = round.(Int, pct .* ni)
    sampcla(x, k, y; seed = seed)
end
