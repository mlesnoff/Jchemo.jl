"""
    samprand(n, k; replace = false)
Build training/test sets by random sampling.  
* `n` : Total nb. observations.
* `k` : Nb. observations to sample (= output `test`).
* `replace` : Boolean. If false (default), the sampling is 
    without replacement.

Two outputs (= row indexes of the data) are returned: 
* `train` (n - `k`),
* `test` (`k`). 

Output `test` is built by random sampling within `1:n`. 

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
    train = zn[setdiff(1:end, s)]
    (train = train, test = s)
end

