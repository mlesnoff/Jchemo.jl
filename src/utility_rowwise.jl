"""
    rowmean(X)
Compute row-wise means of a matrix.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
rowmean(X)
```
""" 
rowmean(X) = vec(Statistics.mean(ensure_mat(X); dims = 2))

"""
    rownorm(X)
Compute row-wise norms of a matrix.
* `X` : Data (n, p).

The norm computed for a row x of `X` is:
* sqrt(x' * x)

Return a vector.

Note: Thanks to @mcabbott 
at https://discourse.julialang.org/t/orders-of-magnitude-runtime-difference-in-row-wise-norm/96363.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)

rownorm(X)
```
""" 
rownorm(X) = sqrt.(vec(sum(abs2, X ; dims = 2)))

"""
    rowstd(X)
Compute row-wise standard deviations (uncorrected) of a matrix`.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
rowstd(X)
```
""" 
rowstd(X) = vec(Statistics.std(ensure_mat(X); dims = 2, corrected = false))

"""
    rowsum(X)
Compute row-wise sums of a matrix.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
using Jchemo
 
X = rand(5, 2) 
rowsum(X)
```
""" 
rowsum(X) = vec(sum(ensure_mat(X); dims = 2))

"""
    rowvar(X)
Compute row-wise variances (uncorrected) of a matrix.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
rowvar(X)
```
""" 
rowvar(X) = vec(Statistics.var(ensure_mat(X); dims = 2, corrected = false))

####### Functions skipping missing data

function rowmeanskip(X)
    X = ensure_mat(X)
    [mean(skipmissing(vrow(X, i))) for i in 1:nro(X)]
end

function rowstdskip(X)
    X = ensure_mat(X)
    [std(skipmissing(vrow(X, i)); corrected = false) 
        for i in 1:nro(X)]
end

function rowsumskip(X)
    X = ensure_mat(X)
    [sum(skipmissing(vrow(X, i))) for i in 1:nro(X)]
end

function rowvarskip(X)
    X = ensure_mat(X)
    [var(skipmissing(vrow(X, i)); corrected = false) 
        for i in 1:nro(X)]
end
