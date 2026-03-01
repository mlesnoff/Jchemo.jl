"""
    matB(X, y, weights::ProbabilityWeights)
Between-class covariance matrix.
* `X` : X-data (n, p).
* `y` : A vector (n) defining the class membership.
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Compute the between-class covariance matrix (output `B`) of `X`. This is the (non-corrected) covariance matrix of 
the weighted class centers.

## Examples
```julia
using Jchemo

n = 20 ; p = 3
X = rand(n, p)
y = rand(1:3, n)
tab(y) 
weights = pweight(ones(n)) 

res = matB(X, y, weights) ;
res.B
res.priors
res.ni
res.lev

res = matW(X, y, weights) ;
res.W
res.Wi

matW(X, y, weights).W + matB(X, y, weights).B
covm(X)

weights = pweight(collect(1:n))
matW(X, y, weights).priors 
matB(X, y, weights).priors 
matW(X, y, weights).W + matB(X, y, weights).B
covm(X, weights)
```
""" 
matB = function(X, y, weights::ProbabilityWeights)
    X = ensure_mat(X)
    y = vec(y)  # required for findall 
    p = nco(X)
    taby = tab(y)
    lev = taby.keys
    ni = taby.vals
    nlev = length(lev)
    priors = aggsumv(weights.values, vec(y)).val   # sub-total weights by class                                
    ct = similar(X, nlev, p)                       # to store class centers
    @inbounds for i in eachindex(lev)
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(vrow(X, s), pweight(weights.values[s]))
    end
    B = covm(ct, pweight(priors))
    (B = B, ct, ni, priors, lev, weights)
end

"""
    matW(X, y, weights::ProbabilityWeights)
Within-class covariance matrices.
* `X` : X-data (n, p).
* `y` : A vector (n) defing the class membership.
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Compute the (non-corrected) within-class and pooled covariance matrices  (outputs `Wi` and `W`, respectively) of `X`. 

If class i contains only one observation, `Wi` is computed by:
* `covm(`X`, `weights`)`.

For examples, see function `matB`. 
""" 
matW = function(X, y, weights::ProbabilityWeights)
    X = ensure_mat(X)
    y = vec(y)  # required for findall 
    p = nco(X) 
    taby = tab(y)
    lev = taby.keys
    ni = taby.vals
    nlev = length(lev)                                 
    priors = aggsumv(weights.values, vec(y)).val     # sub-total weights by class   
    ## When there is at least one class containing only 1 obs, a variable 'Wi_1obs' equal 
    ## to the overal covariance matrix that is then used in the next boucle 'for'.
    ## Another convention could be chosen (e.g., giving weight = 0 to the class(es) with 1 obs), 
    ## but this is not implemented for now.
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
            Wi[i] = covm(X[s, :], pweight(weights.values[s]))
        end
        @. W = W + priors[i] * Wi[i]
        ## Alternative: to give weight = 0 to the class(es) with 1 obs
    end
    (W = W, Wi, ni, priors, lev, weights)
end




