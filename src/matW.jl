"""
    matB(X, y, weights::Weight)
Between-class covariance matrix.
* `X` : X-data (n, p).
* `y` : A vector (n) defining the class membership.
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).

Compute the between-class covariance matrix (output `B`) 
of `X`. This is the (non-corrected) covariance matrix of 
the weighted class centers.

## Examples
```julia
using Jchemo, StatsBase

n = 20 ; p = 3
X = rand(n, p)
y = rand(1:3, n)
tab(y) 
weights = mweight(ones(n)) 

res = matB(X, y, weights) ;
res.B
res.priors
res.ni
res.lev

res = matW(X, y, weights) ;
res.W
res.Wi

matW(X, y, weights).W + matB(X, y, weights).B
cov(X; corrected = false)

v = mweight(collect(1:n))
matW(X, y, v).priors 
matB(X, y, v).priors 
matW(X, y, v).W + matB(X, y, v).B
covm(X, v)
```
""" 
matB = function(X, y, weights::Weight)
    X = ensure_mat(X)
    y = vec(y)  # required for findall 
    p = nco(X)
    taby = tab(y)
    lev = taby.keys
    ni = taby.vals
    nlev = length(lev)
    priors = aggsumv(weights.w, y)      # sub-total weights by class                                
    ct = similar(X, nlev, p)         # class centers
    @inbounds for i in eachindex(lev)
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(vrow(X, s), mweight(weights.w[s]))
    end
    B = covm(ct, mweight(priors))
    (B = B, ct, priors, ni, lev, weights)
end

"""
    matW(X, y, weights::Weight)
Within-class covariance matrices.
* `X` : X-data (n, p).
* `y` : A vector (n) defing the class membership.
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).

Compute the (non-corrected) within-class and pooled covariance 
matrices  (outputs `Wi` and `W`, respectively) of `X`. 

If class i contains only one observation, Wi is computed by:
* `covm(`X`, `weights`)`.

For examples, see function `matB`. 
""" 
matW = function(X, y, weights::Weight)
    X = ensure_mat(X)
    y = vec(y)  # required for findall 
    p = nco(X) 
    taby = tab(y)
    lev = taby.keys
    ni = taby.vals
    nlev = length(lev)                                 
    priors = aggsumv(weights.w, y)     # sub-total weights by class   
    ## Case with at least one class containing only 1 obs:
    ## this creates variable "Wi_1obs" equal to the overal covariance matrix 
    ## (other choices could be chosen) that is then used in the next "for" boucle
    if sum(ni .== 1) > 0
        Wi_1obs = covm(X, weights)
    end
    ## End
    Wi = list(Matrix, nlev)
    W = zeros(eltype(X), p, p)
    @inbounds for i in eachindex(lev) 
        if ni[i] == 1
            Wi[i] = Wi_1obs
        else
            s = findall(y .== lev[i])
            Wi[i] = covm(X[s, :], mweight(weights.w[s]))
        end
        @. W = W + priors[i] * Wi[i]
        ## Alternative: give weight = 0 to the class(es) with 1 obs
    end
    (W = W, Wi, priors, ni, lev, weights)
end




