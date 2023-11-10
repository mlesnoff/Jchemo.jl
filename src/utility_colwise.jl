"""
    colmad(X)
Compute the column-median absolute deviations (MAD) of a matrix.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)

colmad(X)
```
"""
function colmad(X)
    X = ensure_mat(X)
    map(Jchemo.mad, eachcol(X))
end

"""
    colmean(X)
    colmean(X, w)
Compute the column-means of a matrix.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.
    Consider to preliminary normalise `w` to 
    sum to 1 (e.g. function `mweight`).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = mweight(collect(1:n))

colmean(X)
colmean(X, w)
```
""" 
colmean(X) = vec(Statistics.mean(ensure_mat(X); dims = 1))

colmean(X, w) = vec(w' * ensure_mat(X))

"""
    colnorm(X)
    colnorm(X, w)
Compute the column-norms of a matrix.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.
    Consider to preliminary normalise `w` to 
    sum to 1 (e.g. function `mweight`).

The computed norm of a column x of `X` is:
* sqrt(x' * x)

The weighted norm is:
* sqrt(x' * D * x), where D is the diagonal matrix of `w`.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = mweight(collect(1:n))

colnorm(X)
colnorm(X, w)
```
""" 
colnorm(X) = map(norm, eachcol(ensure_mat(X)))

colnorm(X, w) = vec(sqrt.(w' * ensure_mat(X).^2))

"""
    colstd(X)
    colstd(X, w)
Compute the column-standard deviations (uncorrected) of a matrix.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.
    Consider to preliminary normalise `w` to 
    sum to 1 (e.g. function `mweight`).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = mweight(collect(1:n))

colstd(X)
colstd(X, w)
```
""" 
colstd(X) = vec(Statistics.std(ensure_mat(X); corrected = false, dims = 1))

colstd(X, w) = sqrt.(colvar(X, w))

"""
    colsum(X)
    colsum(X, w)
Compute the column-sums of a matrix.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.
    Consider to preliminary normalise `w` to 
    sum to 1 (e.g. function `mweight`).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = mweight(collect(1:n))

colsum(X)
colsum(X, w)
```
""" 
colsum(X) = vec(sum(X; dims = 1))

colsum(X, w) = vec(w' * ensure_mat(X))

"""
    colvar(X)
    colvar(X, w)
Compute the column-variances (uncorrected) of a matrix.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.
    Consider to preliminary normalise `w` to 
    sum to 1 (e.g. function `mweight`).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = mweight(collect(1:n))

colvar(X)
colvar(X, w)
```
""" 
colvar(X) = vec(Statistics.var(ensure_mat(X); corrected = false, dims = 1))

function colvar(X, w)
    X = ensure_mat(X)
    v = colmean(X, w)
    colnorm(X .- v', w).^2
end

####### Functions skipping missing data

function colmeanskip(X)
    X = ensure_mat(X)
    [mean(skipmissing(vcol(X, i))) for i in 1:nco(X)]
end

function colstdskip(X)
    X = ensure_mat(X)
    [std(skipmissing(vcol(X, i)); corrected = false) for i in 1:nco(X)]
end

function colsumskip(X)
    X = ensure_mat(X)
    [sum(skipmissing(vcol(X, i))) for i in 1:nco(X)]
end

function colvarskip(X)
    X = ensure_mat(X)
    [var(skipmissing(vcol(X, i)); corrected = false) for i in 1:nco(X)]
end

## With weights
function colmeanskip(X, w)
    X = ensure_mat(X)
    p = nco(X)
    w = collect(w) # rmrow does not accept UnitRange
    z = zeros(p)
    for i = 1:p
        s = ismissing.(vcol(X, i))
        w = mweight(rmrow(w, s))
        z[i] = sum(w .* rmrow(X[:, i], s))
    end
    z
end

colsumskip(X, w) = colmeanskip(X, w)

function colvarskip(X, w)
    X = ensure_mat(X)
    p = nco(X)
    w = collect(w)
    z = colmeanskip(X, w)
    @inbounds for i = 1:p
        s = ismissing.(vcol(X, i))
        w = mweight(rmrow(w, s))
        z[i] = dot(w, (rmrow(X[:, i], s) .- z[i]).^2)        
    end
    z 
end

colstdskip(X, w) = sqrt.(colvarskip(X, w))

