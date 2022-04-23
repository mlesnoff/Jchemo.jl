"""
    matB(X, y; fun = mean)
Compute the between covariance matrix ("B") of `X`.
* `X` : X-data (n, p).
* `y` : A vector (n) defing the class memberships.

## Examples
```julia
n = 10 ; p = 3
X = rand(n, p)
X
y = sample(1:3, n; replace = true)
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
    y = vec(y) 
    z = aggstat(X; group = y, fun = mean)
    B = covm(z.X, mweight(z.ni))
    (B = B, ct = z.X, lev = z.lev, ni = z.ni)
end


"""
    matW(X, y; fun = mean)
Compute the within covariance matrix ("W") of `X`.
* `X` : X-data (n, p).
* `y` : A vector (n) defing the class memberships.

If class "i" contains only one observation, 
W_i is computed as `cov(X; corrected = false)`.

For examples, see `?matB`. 
""" 
matW = function(X, y)
    X = ensure_mat(X)
    y = vec(y)  
    ztab = tab(y)
    lev = ztab.keys
    nlev = length(lev)
    ni = collect(values(ztab))
    # Case with y(s) with only 1 obs
    sum(ni .== 1) > 0 ? sigma_1obs = cov(X; corrected = false) : nothing
    # End
    w = mweight(ni)
    Wi = list(nlev, Matrix{Float64})
    W = zeros(1, 1)
    @inbounds for i in 1:nlev 
        if ni[i] == 1
            Wi[i] = sigma_1obs
        else
            s = findall(y .== lev[i])
            Wi[i] = cov(X[s, :]; corrected = false)
        end
        if i == 1  
            W = w[i] * Wi[i] 
        else 
            W = W + w[i] * Wi[i]
            # Alternative: Could give weight=0 to the class(es) with 1 obs
        end
    end
    (W = W, Wi = Wi, lev = lev, ni = ni)
end






