"""
    sampdp(X, k::Int; metric::Symbol = :eucl)
Build training vs. test sets by DUPLEX sampling.  
* `X` : X-data (n, p).
* `k` : Nb. pairs (training/test) of observations to sample. Must be <= n / 2. 
Keyword arguments:
* `metric` : Metric used for the distance computation. Possible values are: `:eucl` (Euclidean), 
    `:mah` (Mahalanobis).

Three outputs (= row indexes of the data) are returned: 
* `train` (`k`), 
* `test` (`k`),
* `remain` (n - 2 * `k`). 

Outputs `train` and `test` are built from the DUPLEX algorithm (Snee, 1977 p.421). They are expected 
to cover approximately the same X-space region and have similar statistical properties. 

In practice, when output `remain` is not empty (i.e. when there are remaining observations), one common strategy 
is to add it to output `train`.

## References
Kennard, R.W., Stone, L.A., 1969. Computer aided design of experiments. Technometrics, 11(1), 137-148.

Snee, R.D., 1977. Validation of Regression Models: Methods and Examples. Technometrics 19, 415-428. 
https://doi.org/10.1080/00401706.1977.10489581

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
function sampdp(X, k::Int; metric::Symbol = :eucl)
    @assert in([:eucl, :mah, :sam, :cos, :cor])(metric) "Wrong value for argument 'metric'."
    X = ensure_mat(X)
    if metric == :eucl
        D = eucl2(X, X)
    else
        D = mah2chol(X, X)
    end
    n = nro(D)
    vn = 1:n
    ## Initial selection of 2 pairs of obs.
    s = findall(D .== maximum(D)) 
    s1 = [s[1][1] ; s[1][2]]
    vD = copy(D)
    vD[s1, :] .= -Inf ;
    vD[:, s1] .= -Inf ;
    s = findall(D .== maximum(vD)) 
    s2 = [s[1][1] ; s[1][2]]
    ## Candidates
    candidat = vn[setdiff(1:end, [s1 ; s2])]
    @inbounds for i = 1:(k - 2)
        u = vec(minimum(D[s1, candidat], dims = 1))
        zs = candidat[findall(u .== maximum(u))[1]]
        s1 = [s1 ; zs]
        candidat = vn[setdiff(1:end, s1)]
        u = vec(minimum(D[s2, candidat], dims = 1))
        zs = candidat[findall(u .== maximum(u))[1]]
        s2 = [s2 ; zs]
        candidat = vn[setdiff(1:end, [s1 ; s2])]
    end
    sort!(s1)
    sort!(s2)
    (train = s1, test = s2, remain = candidat)
end
    
