"""
    matB(X, y, weights = ones(nro(X)))
Between-class covariance matrix.
* `X` : X-data (n, p).
* `y` : A vector (n) defining the class membership.
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.

Compute the between-class covariance matrix (B) of `X`.
This is the (non-corrected) covariance matrix of 
the weighted class centers.

## Examples
```julia
n = 20 ; p = 3
X = rand(n, p)
X
y = rand(1:3, n)
res = matB(X, y)
res.B
res.lev
res.ni

res = matW(X, y)
res.W 
res.Wi

matW(X, y).W + matB(X, y).B 
cov(X; corrected = false)

w = ones(n)
matW(X, y, w).W + matB(X, y, w).B
cov(X; corrected = false)

w = rand(n)
matW(X, y, w).theta 
matB(X, y, w).theta 

matW(X, y, w).W + matB(X, y, w).B
covm(X, w)
```
""" 
matB = function(X, y, weights = ones(nro(X)))
    X = ensure_mat(X)
    weights = mweight(weights)
    p = nco(X)
    ztab = tab(y)
    lev = ztab.keys
    ni = ztab.vals
    nlev = length(lev)
    theta = aggstat(weights, y; fun = sum).X
    ct = similar(X, nlev, p)
    for i = 1:nlev
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(X[s, :], weights[s])
    end
    B = covm(ct, theta)
    (B = B, ct, theta, weights, lev = lev, ni)
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
    weights = mweight(weights)
    p = nco(X) 
    ztab = tab(y)
    lev = ztab.keys
    ni = ztab.vals
    nlev = length(lev)
    theta = aggstat(weights, y; fun = sum).X
    ## Case with at least one class with only 1 obs:
    ## this creates Wi_1obs used in the boucle
    if sum(ni .== 1) > 0
        Wi_1obs = covm(X, weights)
    end
    ## End
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
    (W = W, Wi, theta, weights, lev, ni)
end




