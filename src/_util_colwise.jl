"""
    colsum(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    colsum(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
Column-wise sums of a matrix.
* `X` : Matrix (n, p).
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
function colsum(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    s = zeros(eltype(X), nco(X))
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            s[j] += X[i, j]
        end
    end
    s
end

function colsum(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
    s = zeros(eltype(X), nco(X))
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            s[j] += X[i, j] * weights.values[i]
        end
    end
    s
end

"""
    colmean(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    colmean(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
Column-wise means of a matrix.
* `X` : Matrix (n, p).
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
colmean(X::AbstractMatrix{Q}) where Q <: AbstractFloat = colsum(X) / nro(X)

colmean(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat = colsum(X, weights)

"""
    colnorm(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    colnorm(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
Column-wise norms of a matrix.
* `X` : Matrix (n, p).
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
colnorm(X::AbstractMatrix{Q}) where Q <: AbstractFloat = sqrt.(colnorm2(X))

colnorm(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat = sqrt.(colnorm2(X, weights))

"""
    colnorm2(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    colnorm2(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
Column-wise squared norms of a matrix.
* `X` : Matrix (n, p).
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
function colnorm2(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    s = zeros(eltype(X), nco(X))
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            s[j] += X[i, j]^2
        end
    end
    s
end

function colnorm2(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
    s = zeros(eltype(X), nco(X))
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            s[j] += X[i, j]^2 * weights.values[i]
        end
    end
    s
end

"""
    colvar(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    colvar(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
Column-wise (uncorrected) variances of a matrix.
* `X` : Matrix (n, p).
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
function colvar(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    s = similar(X, nco(X))
    Threads.@threads for j in axes(X, 2)
        s[j] = varv(vcol(X, j))
    end
    s
end

function colvar(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
    s = similar(X, nco(X))
    Threads.@threads for j in axes(X, 2)
        s[j] = varv(vcol(X, j), weights)
    end
    s
end

"""
    colstd(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    colstd(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
Column-wise (uncorrected) standard deviations of a matrix.
* `X` : Matrix (n, p).
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
colstd(X::AbstractMatrix{Q}) where Q <: AbstractFloat = sqrt.(colvar(X))

colstd(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat = sqrt.(colvar(X, weights))

"""
    colprt(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    colprt(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
Column-wise (uncorrected) standard deviations of a matrix.
* `X` : Matrix (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

Return a vector (p).

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
w = pweight(rand(n))

colprt(X)
colprt(X, w)
```
""" 
colprt(X::AbstractMatrix{Q}) where Q <: AbstractFloat = sqrt.(colstd(X))

colprt(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat = sqrt.(colstd(X, weights))

"""
    colmed(X::AbstractMatrix{Q}) where Q <: AbstractFloat
Column-wise medians of a matrix.
* `X` : Matrix (n, p).

Return a vector (p).

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)

colmed(X)
```
""" 
function colmed(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    s = similar(X, nco(X))
    Threads.@threads for j in axes(X, 2)
        s[j] = Statistics.median(vcol(X, j))
    end
    s
end

"""
    colmad(X::AbstractMatrix{Q}) where Q <: AbstractFloat
Column-wise median absolute deviations (MAD) of a matrix.
* `X` : Matrix (n, p).

Return a vector (p).

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)

colmad(X)
```
"""
function colmad(X::AbstractMatrix{Q}) where Q <: AbstractFloat
    s = similar(X, nco(X))
    Threads.@threads for j in axes(X, 2)
        s[j] = madv(vcol(X, j))
    end
    s
end

colmad(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat = colmad(X)  # for consistency when weights

"""
    def_colscal(scal::Symbol = :std)
Define the function of column scaling.
* `scal` : Symbol defining the scaling. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).
```
"""
function def_colscal(scal::Symbol = :std)
    dict = Dict(
        :std => colstd, 
        :prt => colprt,
        :mad => colmad
        )
    dict[scal]
end

##### Functions skipping missing data

colsumskip(X) = [Base.sum(skipmissing(x)) for x in eachcol(ensure_mat(X))]
function colsumskip(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
    X = ensure_mat(X)
    v = zeros(Q, nco(X))
    @inbounds for j in axes(X, 2)
        s = ismissing.(vcol(X, j))
        w = pweight(rmrow(weights.values, s))
        v[j] = sum(w.values .* rmrow(X[:, j], s))
    end
    v
end

colmeanskip(X) = [Statistics.mean(skipmissing(x)) for x in eachcol(ensure_mat(X))]
colmeanskip(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat = colsumskip(X, weights)

colstdskip(X) = [Statistics.std(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]
colstdskip(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat = sqrt.(colvarskip(X, weights))

colvarskip(X) = [Statistics.var(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]
function colvarskip(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}) where Q <: AbstractFloat
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



