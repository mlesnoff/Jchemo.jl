"""
    matB(X, y)
Between-class covariance matrix (B).
* `X` : X-data (n, p).
* `y` : A vector (n) defining the class membership.

Compute the between-class covariance matrix (B) of `X`.
This is the (non-corrected) covariance matrix of 
the weighted (ni / n) class centers.

## Examples
```julia
n = 10 ; p = 3
X = rand(n, p)
X
y = rand(1:3, n)
#y = [3 ; ones(n - 2) ; 10]
res = matB(X, y)
res.B
res.lev
res.ni

res = matW(X, y)
res.W 
res.Wi

matW(X, y).W + matB(X, y).B 
cov(X; corrected = false)
```
""" 
matB = function(X, y)
    X = ensure_mat(X)
    res = aggstat(X, y; fun = mean)
    ni = tab(y).vals
    B = covm(res.X, mweight(ni))
    (B = B, ct = res.X, lev = res.lev, ni)
end

"""
    matW(X, y, weights = ones(nro(X)))
Within-class covariance matrices.
* `X` : X-data (n, p).
* `y` : A vector (n) defing the class membership.
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.

Compute the (non-corrected) within-class covariance matrices (Wi)
 of `X`, and the pooled covariance matrix W by:
* W = (n1 / n) * W1 + ... + (nI / n) * WI 

If class i contains only one observation, 
Wi is computed by `cov(`X`; corrected = false)`.

For examples, see `?matB`. 
""" 
matW = function(X, y, weights = ones(nro(X)))
    X = ensure_mat(X)
    y = vec(y)  # required for findall 
    p = nco(X) 
    ztab = tab(y)
    lev = ztab.keys
    ni = ztab.vals
    nlev = length(lev)
    weights = mweight(weights)
    ## Case with at least one class with only 1 obs:
    ## this creates Wi_1obs used in the boucle
    if sum(ni .== 1) > 0
        Wi_1obs = covm(X, weights)
    end
    ## End
    theta = mweight(ni)
    Wi = list(nlev, Matrix{Float64})
    W = zeros(p, p)
    @inbounds for i in 1:nlev 
        if ni[i] == 1
            Wi[i] = Wi_1obs
        else
            s = findall(y .== lev[i])
            Wi[i] = covm(X[s, :], weights[s])
        end
        @. W = W + theta[i] * Wi[i]
        ## Alternative: give weight = 0 to the class(es) with 1 obs
    end
    (W = W, Wi, weights, lev, ni)
end




