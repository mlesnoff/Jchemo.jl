"""
    samprand(n, k; replace = false)
Split training/test sets by random sampling.  
* `n` : Total nb. observations.
* `k` : Nb. observations to sample (output `train`).
* `replace` : Boolean. If false (default), the sampling is without 
    replacement.

Two outputs (indexes) are returned: 
* `train` (`k`),
* `test` (`n` - `k`). 

Output `train` is built by random sampling within `1:n`, with no replacement. 

## Examples
```julia
n = 10
samprand(n, 7)
```
""" 
function samprand(n, k; replace = false)
    k = Int(round(k))
    zn = collect(1:n)
    s = sample(zn, k; replace = replace)
    sort!(s)
    test = zn[setdiff(1:end, s)]
    (train = s, test = test)
end

