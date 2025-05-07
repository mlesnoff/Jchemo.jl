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
rowmean(X) = colmean(ensure_mat(X)')

"""
    rownorm(X)
Compute row-wise norms of a matrix.
* `X` : Data (n, p).

The norm of each row x of `X` is computed as:
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
rownorm(X) = colnorm(ensure_mat(X)')

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
rowstd(X) = colstd(ensure_mat(X)')

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
rowsum(X) = colsum(ensure_mat(X)')

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
rowvar(X) = rowvar(ensure_mat(X)')

##### Functions skipping missing data
rowmeanskip(X) = colmeanskip(ensure_mat(X)')
rowstdskip(X) = colstdskip(ensure_mat(X)')
rowsumskip(X) = colsumskip(ensure_mat(X)')
rowvarskip(X) = colvarskip(ensure_mat(X)')

