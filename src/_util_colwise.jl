"""
    colsum(X)
    colsum(X, weights::ProbabilityWeights)
Column-wise sums of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector (p).

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
    Q = eltype(X)
    n, p = size(X)
    s = zeros(Q, p)
    Threads.@threads for j = 1:p
        @inbounds for i in 1:n
            s[j] += X[i, j]
        end
    end
    s
end

function colsum(X, weights::ProbabilityWeights)
    X = ensure_mat(X)
    Q = eltype(X)
    n, p = size(X)
    s = zeros(Q, p)
    Threads.@threads for j = 1:p
        @inbounds for i in 1:n
            s[j] += X[i, j] * weights.values[i]
        end
    end
    s
end

"""
    colmean(X)
    colmean(X, weights::ProbabilityWeights)
Column-wise means of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector (p).

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
Column-wise norms of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector (p).

The norm of each column x of `X` is computed by:
* sqrt(x' * x)

The weighted norm is:
* sqrt(x' * D * x), where D is the diagonal matrix of `weights.values`

**Warning:** `colnorm(X, pweight(ones(n)))` = `colnorm(X) / sqrt(n)`.

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
colnorm(X) = sqrt.(colnorm2(X))

colnorm(X, weights::ProbabilityWeights) = sqrt.(colnorm2(X, weights))

"""
    colnorm2(X)
    colnorm2(X, weights::ProbabilityWeights)
Column-wise squared norms of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

See function `colnorm`.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = pweight(rand(n))

colnorm2(X)
colnorm2(X, w)
```
"""
function colnorm2(X)
    X = ensure_mat(X)
    Q = eltype(X)
    n, p = size(X)
    s = zeros(Q, p)
    Threads.@threads for j = 1:p
        @inbounds for i in 1:n
            s[j] += X[i, j]^2
        end
    end
    s
end

function colnorm2(X, weights::ProbabilityWeights)
    X = ensure_mat(X)
    Q = eltype(X)
    n, p = size(X)
    s = zeros(Q, p)
    Threads.@threads for j = 1:p
        @inbounds for i in 1:n
            s[j] += X[i, j]^2 * weights.values[i]
        end
    end
    s
end

"""
    colstd(X)
    colstd(X, weights::ProbabilityWeights)
Column-wise (uncorrected) standard deviations of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector (p).

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
colstd(X) = sqrt.(colvar(X))

colstd(X, weights::ProbabilityWeights) = sqrt.(colvar(X, weights))

"""
    colvar(X)
    colvar(X, weights::ProbabilityWeights)
Column-wise (uncorrected) variances of a matrix.
* `X` : Data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector (p).

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
Column-wise medians of a matrix.
* `X` : Data (n, p).

Return a vector (p).

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
Column-wise median absolute deviations (MAD) of a matrix.
* `X` : Data (n, p).

Return a vector (p).

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
function colsumskip(X, weights::ProbabilityWeights)
    X = ensure_mat(X)
    p = nco(X)
    v = zeros(p)
    @inbounds for j = 1:p
        s = ismissing.(vcol(X, j))
        w = pweight(rmrow(weights.values, s))
        v[j] = sum(w.values .* rmrow(X[:, j], s))
    end
    v
end
colmeanskip(X) = [Statistics.mean(skipmissing(x)) for x in eachcol(ensure_mat(X))]
colmeanskip(X, weights::ProbabilityWeights) = colsumskip(X, weights)
colstdskip(X) = [Statistics.std(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]
colstdskip(X, weights::ProbabilityWeights) = sqrt.(colvarskip(X, weights))
colvarskip(X) = [Statistics.var(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]
function colvarskip(X, weights::ProbabilityWeights)
    X = ensure_mat(X)
    p = nco(X)
    v = colmeanskip(X, weights)
    @inbounds for j = 1:p
        s = ismissing.(vcol(X, j))
        w = pweight(rmrow(weights.values, s))
        v[j] = dot(w.values, (rmrow(X[:, j], s) .- v[j]).^2)        
    end
    v 
end

