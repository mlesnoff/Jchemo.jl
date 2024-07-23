"""
    sampdp(X, k::Int; metric = :eucl)
Build training vs. test sets by DUPLEX sampling.  
* `X` : X-data (n, p).
* `k` : Nb. pairs (training/test) of observations 
    to sample. Must be <= n / 2. 
Keyword arguments:
* `metric` : Metric used for the distance computation.
    Possible values are: `:eucl` (Euclidean), 
    `:mah` (Mahalanobis).

Three outputs (= row indexes of the data) are returned: 
* `train` (`k`), 
* `test` (`k`),
* `remain` (n - 2 * `k`). 

Outputs `train` and `test` are built from the DUPLEX algorithm 
(Snee, 1977 p.421). They are expected to cover approximately the same 
X-space region and have similar statistical properties. 

In practice, when output `remain` is not empty (i.e. when there 
are remaining observations), one common strategy is to add 
it to output `train`.

## References
Kennard, R.W., Stone, L.A., 1969. Computer aided design of experiments. 
Technometrics, 11(1), 137-148.

Snee, R.D., 1977. Validation of Regression Models: Methods and Examples. 
Technometrics 19, 415-428. https://doi.org/10.1080/00401706.1977.10489581

## Examples
```julia
using Jchemo

X = [0.381392  0.00175002 ; 0.1126    0.11263 ; 
    0.613296  0.152485 ; 0.726536  0.762032 ;
    0.367451  0.297398 ; 0.511332  0.320198 ; 
    0.018514  0.350678] 

k = 3
sampdp(X, k)
```
""" 
function sampdp(X, k::Int; metric = :eucl)
    @assert in([:eucl, :mah])(metric) "Wrong value for argument 'metric'."
    if metric == :eucl
        D = euclsq(X, X)
    else
        D = mahsqchol(X, X)
    end
    n = size(D, 1)
    zn = 1:n
    ## Initial selection of 2 pairs of obs.
    s = findall(D .== maximum(D)) 
    s1 = [s[1][1] ; s[1][2]]
    zD = copy(D)
    zD[s1, :] .= -Inf ;
    zD[:, s1] .= -Inf ;
    s = findall(D .== maximum(zD)) 
    s2 = [s[1][1] ; s[1][2]]
    ## Candidates
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
    
