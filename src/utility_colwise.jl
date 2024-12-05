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
colmad(X) = map(Jchemo.mad, eachcol(ensure_mat(X)))

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

colmean(X, weights::Weight) = vec(weights.w' * ensure_mat(X))  # = colsum

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
colmed(X) = map(Statistics.median, eachcol(ensure_mat(X)))

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
* Warning: `colnorm(X, mweight(ones(n))) = colnorm(X) / sqrt(n)`

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
colnorm(X) = map(normv, eachcol(ensure_mat(X)))

colnorm(X, weights::Weight) = sqrt.(vec(weights.w' * ensure_mat(X).^2))

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
colstd(X) = map(v -> Statistics.std(v ; corrected = false), eachcol(ensure_mat(X)))

colstd(X, weights::Weight) = colnorm(X .- colmean(ensure_mat(X), weights)', weights)

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
    r = similar(X, p)
    Threads.@threads for j = 1:p
        r[j] = sum(vcol(X, j))
    end
    r
end

colsum(X, weights::Weight) = vec(weights.w' * ensure_mat(X))

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
colvar(X) = map(v -> Statistics.var(v ; corrected = false), eachcol(ensure_mat(X)))

colvar(X, weights::Weight) = colstd(X, weights).^2

###### Functions skipping missing data

colmeanskip(X) = [Statistics.mean(skipmissing(x)) for x in eachcol(ensure_mat(X))]

colstdskip(X) = [Statistics.std(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]

colsumskip(X) = [Base.sum(skipmissing(x)) for x in eachcol(ensure_mat(X))]

colvarskip(X) = [Statistics.var(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]

## With weights
function colmeanskip(X, weights::Weight)
    X = ensure_mat(X)
    p = nco(X)
    z = zeros(p)
    for j = 1:p
        s = ismissing.(vcol(X, j))
        zw = mweight(rmrow(weights.w, s)).w
        z[j] = sum(zw .* rmrow(X[:, j], s))
    end
    z
end

colsumskip(X, weights::Weight) = colmeanskip(X, weights)

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

colstdskip(X, weights::Weight) = sqrt.(colvarskip(X, weights))

