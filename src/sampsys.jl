"""
    sampsys(y, k::Int)
Build training vs. test sets by systematic sampling over a quantitative variable.  
* `y` : Quantitative variable (n) to sample.
* `k` : Nb. test observations to sample. Must be >= 2.

Two outputs are returned (= row indexes of the data): 
* `train` (n - `k`),
* `test` (`k`). 

Output `test` is built by systematic sampling over the rank of the `y` observations. For instance 
if `k` / n ~ .3, one observation over three observations over the sorted `y` is selected. 

Output `test` always contains the indexes of the minimum and maximum of `y`.

## Examples
```julia
using Jchemo 

y = rand(7)
[y sort(y)]
res = sampsys(y, 3)
sort(y[res.test])
```
""" 
function sampsys(y, k::Int)
    y = vec(y)
    n = length(y)
    nint = k - 1            # nb. intervals
    alpha = (n - 1) / nint  # step
    z = collect(1:alpha:n)
    z = Int.(round.(z))
    z = unique(z)
    id = sortperm(y)
    zn = collect(1:n)
    s = zn[id[z]]
    sort!(s)
    train = zn[setdiff(1:end, s)]
    (train = train, test = s)
end


