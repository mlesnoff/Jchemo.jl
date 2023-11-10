"""
    rowmean(X)
Compute the row-means of a matrix.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
rowmean(X)
```
""" 
rowmean(X) = vec(Statistics.mean(ensure_mat(X); dims = 2))

"""
    rowstd(X)
Compute the row-standard deviations (uncorrected) of a matrix`.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
rowstd(X)
```
""" 
rowstd(X) = vec(Statistics.std(ensure_mat(X); dims = 2, corrected = false))

"""
    rowsum(X)
Compute the row-sums of a matrix.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
X = rand(5, 2) 
rowsum(X)
```
""" 
rowsum(X) = vec(sum(ensure_mat(X); dims = 2))

"""
    rowvar(X)
Compute the row-variances (uncorrected) of a matrix.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
rowvar(X)
```
""" 
rowvar(X) = vec(Statistics.var(ensure_mat(X); dims = 2, corrected = false))

####### Functions with skip missing data

function rowmeanskip(X)
    X = ensure_mat(X)
    [mean(skipmissing(vrow(X, i))) for i in 1:nro(X)]
end

function rowstdskip(X)
    X = ensure_mat(X)
    [std(skipmissing(vrow(X, i)); corrected = false) for i in 1:nro(X)]
end

function rowsumskip(X)
    X = ensure_mat(X)
    [sum(skipmissing(vrow(X, i))) for i in 1:nro(X)]
end

function rowvarskip(X)
    X = ensure_mat(X)
    [var(skipmissing(vrow(X, i)); corrected = false) for i in 1:nro(X)]
end
