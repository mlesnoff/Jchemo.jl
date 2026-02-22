"""
    colsum(X)
    colsum(X, weights::ProbabilityWeights)
Compute column-wise sums of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = pweight(rand(n))

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

colsum(X, weights::ProbabilityWeights) = vec(weights.values' * ensure_mat(X))

"""
    colmean(X)
    colmean(X, weights::ProbabilityWeights)
Compute column-wise means of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = pweight(rand(n))

colmean(X)
colmean(X, w)
```
""" 
colmean(X) = colsum(X) / nro(X)

colmean(X, weights::ProbabilityWeights) = colsum(X, weights)

"""
    colnorm(X)
    colnorm(X, weights::ProbabilityWeights)
Compute column-wise norms of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

The norm of each column x of `X` is computed by:
* sqrt(x' * x)

The weighted norm is:
* sqrt(x' * D * x), where D is the diagonal matrix of `weights.values`

**Warning:** `colnorm(X, pweight(ones(n)))` = `colnorm(X) / sqrt(n)`.

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = pweight(rand(n))

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

function colnorm(X, weights::ProbabilityWeights)
    X = ensure_mat(X) 
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = normv(vcol(X, j), weights)
    end
    s
end

## Not exported
function colnorm2(X)
    X = ensure_mat(X)
    p = nco(X)
    s = similar(X, p)
    Threads.@threads for j = 1:p
        s[j] = dot(vcol(X, j), vcol(X, j))
    end
    s
end

"""
    colstd(X)
    colstd(X, weights::ProbabilityWeights)
Compute column-wise standard deviations (uncorrected) of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = pweight(rand(n))

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

function colstd(X, weights::ProbabilityWeights)
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
    colvar(X, weights::ProbabilityWeights)
Compute column-wise variances (uncorrected) of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = pweight(rand(n))

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

function colvar(X, weights::ProbabilityWeights)
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
function colsumskip(X, weights::ProbabilityWeights)
    X = ensure_mat(X)
    p = nco(X)
    z = zeros(p)
    @inbounds for j = 1:p
        s = ismissing.(vcol(X, j))
        zw = pweight(rmrow(weights.values, s)).values
        z[j] = sum(zw .* rmrow(X[:, j], s))
    end
    z
end
colmeanskip(X, weights::ProbabilityWeights) = colsumskip(X, weights)
colstdskip(X, weights::ProbabilityWeights) = sqrt.(colvarskip(X, weights))
function colvarskip(X, weights::ProbabilityWeights)
    X = ensure_mat(X)
    p = nco(X)
    z = colmeanskip(X, weights)
    @inbounds for j = 1:p
        s = ismissing.(vcol(X, j))
        zw = pweight(rmrow(weights.values, s)).values
        z[j] = dot(zw, (rmrow(X[:, j], s) .- z[j]).^2)        
    end
    z 
end

