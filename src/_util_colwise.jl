"""
    colsum(X)
    colsum(X, weights::Weight)
Compute column-wise sums of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = mweight(rand(n))

colsum(X)
colsum(X, w)
```
""" 
function colsum(X)
    X = ensure_mat(X)
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = sumv(vcol(X, j))
    end
    s
end

colsum(X, weights::Weight) = vec(weights.w' * ensure_mat(X))

"""
    colmean(X)
    colmean(X, weights::Weight)
Compute column-wise means of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = mweight(rand(n))

colmean(X)
colmean(X, w)
```
""" 
colmean(X) = colsum(X) / nro(X)

colmean(X, weights::Weight) = colsum(X, weights)

"""
    colnorm(X)
    colnorm(X, weights::Weight)
Compute column-wise norms of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).

The norm computed for a column x of `X` is:
* sqrt(x' * x)

The weighted norm is:
* sqrt(x' * D * x), where D is the diagonal matrix of `weights.w`
**Warning:** `colnorm(X, mweight(ones(n)))` = `colnorm(X) / sqrt(n)`.

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = mweight(rand(n))

colnorm(X)
colnorm(X, w)
```
""" 
function colnorm(X)
    X = ensure_mat(X)
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = normv(vcol(X, j))
    end
    s
end

function colnorm(X, weights::Weight)
    X = ensure_mat(X) 
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = normv(vcol(X, j), weights)
    end
    s
end

"""
    colstd(X)
    colstd(X, weights::Weight)
Compute column-wise standard deviations (uncorrected) of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = mweight(rand(n))

colstd(X)
colstd(X, w)
```
""" 
function colstd(X)
    X = ensure_mat(X)
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = stdv(vcol(X, j))
    end
    s
end

function colstd(X, weights::Weight)
    X = ensure_mat(X) 
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = stdv(vcol(X, j), weights)
    end
    s
end

"""
    colvar(X)
    colvar(X, weights::Weight)
Compute column-wise variances (uncorrected) of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = mweight(rand(n))

colvar(X)
colvar(X, w)
```
""" 
function colvar(X)
    X = ensure_mat(X)
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = varv(vcol(X, j))
    end
    s
end

function colvar(X, weights::Weight)
    X = ensure_mat(X) 
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = varv(vcol(X, j), weights)
    end
    s
end

"""
    colmed(X)
Compute column-wise medians of a matrix.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)

colmed(X)
```
""" 
function colmed(X)
    X = ensure_mat(X)
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = Statistics.median(vcol(X, j))
    end
    s
end

"""
    colmad(X)
Compute column-wise median absolute deviations (MAD) of a matrix.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)

colmad(X)
```
"""
function colmad(X)
    X = ensure_mat(X)
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = madv(vcol(X, j))
    end
    s
end

##### Functions skipping missing data
colsumskip(X) = [Base.sum(skipmissing(x)) for x in eachcol(ensure_mat(X))]
colmeanskip(X) = [Statistics.mean(skipmissing(x)) for x in eachcol(ensure_mat(X))]
colstdskip(X) = [Statistics.std(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]
colvarskip(X) = [Statistics.var(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]
## With weights
function colsumskip(X, weights::Weight)
    X = ensure_mat(X)
    p = nco(X)
    z = zeros(p)
    @inbounds for j = 1:p
        s = ismissing.(vcol(X, j))
        zw = mweight(rmrow(weights.w, s)).w
        z[j] = sum(zw .* rmrow(X[:, j], s))
    end
    z
end
colmeanskip(X, weights::Weight) = colsumskip(X, weights)
colstdskip(X, weights::Weight) = sqrt.(colvarskip(X, weights))
function colvarskip(X, weights::Weight)
    X = ensure_mat(X)
    p = nco(X)
    z = colmeanskip(X, weights)
    @inbounds for j = 1:p
        s = ismissing.(vcol(X, j))
        zw = mweight(rmrow(weights.w, s)).w
        z[j] = dot(zw, (rmrow(X[:, j], s) .- z[j]).^2)        
    end
    z 
end

