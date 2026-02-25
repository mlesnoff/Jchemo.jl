"""
    rowsum(X)
Row-wise sums of a matrix.
* `X` : Data (n, p).

Return a vector (n).

## Examples
```julia
using Jchemo
 
X = rand(5, 2) 
rowsum(X)
```
""" 
rowsum(X) = colsum(ensure_mat(X)')

"""
    rowmean(X)
Row-wise means of a matrix.
* `X` : Data (n, p).

Return a vector (n).

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
Row-wise norms of a matrix.
* `X` : Data (n, p).

Return a vector (n).

The norm of each row x of `X` is computed as:
* sqrt(x' * x)

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)

rownorm(X)
```
""" 
rownorm(X) = sqrt.(rownorm2(X))

"""
    rownorm2(X)
Row-wise squared norms of a matrix.
* `X` : Data (n, p).

Return a vector (n).

See function `rownorm`.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)

rownorm2(X)
```
""" 
rownorm2(X) = colnorm2(ensure_mat(X)')

"""
    rowstd(X)
Row-wise (uncorrected) standard deviations of a matrix`.
* `X` : Data (n, p).

Return a vector (n).

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
rowstd(X)
```
""" 
rowstd(X) = sqrt.(rowvar(X))

"""
    rowvar(X)
Row-wise (uncorrected) variances of a matrix.
* `X` : Data (n, p).

Return a vector (n).

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
rowvar(X)
```
""" 
rowvar(X) = colvar(ensure_mat(X)')

##### Functions skipping missing data
rowsumskip(X) = colsumskip(ensure_mat(X)')
rowmeanskip(X) = colmeanskip(ensure_mat(X)')
rowstdskip(X) = colstdskip(ensure_mat(X)')
rowvarskip(X) = colvarskip(ensure_mat(X)')

