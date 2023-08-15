"""
    rowmean(X)
Compute the mean of each row of `X`.
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
Compute the (uncorrected) standard deviation of each row of `X`.
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
Compute the sum of each row of `X`.
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
Compute the (uncorrected) standard deviation of each row of `X`.
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

####### SKIP MISSING

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
