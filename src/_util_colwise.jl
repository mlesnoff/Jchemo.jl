"""
    colsum(X::DataFrame)
    colsum(X::AbstMatVec{Q}) where Q <: Union{Signed, Float}
    colsum(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Union{Signed, Float}
Column-wise sums of a matrix.
* `X` : Matrix (n, p) or vector (n).
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
colsum(X::DataFrame) = colsum(ensure_mat(X))

function colsum(X::AbstMatVec{Q}) where Q <: Union{Signed, Float}
    s = zeros(eltype(X), nco(X))
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            s[j] += X[i, j]
        end
    end
    s
end

function colsum(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Union{Signed, Float}
    s = zeros(eltype(X), nco(X))
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            s[j] += X[i, j] * weights.values[i]
        end
    end
    s
end

"""
    colmean(X::DataFrame)
    colmean(X::AbstMatVec{Q}) where Q <: Float
    colmean(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float
Column-wise means of a matrix.
* `X` : Matrix (n, p) or vector (n).
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
colmean(X::DataFrame) = colmean(ensure_mat(X))

colmean(X::AbstMatVec{Q}) where Q <: Float = colsum(X) / nro(X)

colmean(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float = colsum(X, weights)

"""
    colnorm(X::DataFrame)
    colnorm(X::AbstMatVec{Q}) where Q <: Float
    colnorm(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float
Column-wise norms of a matrix.
* `X` : Matrix (n, p) or vector (n).
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
colnorm(X::DataFrame) = colnorm(ensure_mat(X))

colnorm(X::AbstMatVec{Q}) where Q <: Float = sqrt.(colnorm2(X))

colnorm(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float = sqrt.(colnorm2(X, weights))

"""
    colnorm2(X::DataFrame)
    colnorm2(X::AbstMatVec{Q}) where Q <: Float
    colnorm2(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float
Column-wise squared norms of a matrix.
* `X` : Matrix (n, p) or vector (n).
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
colnorm2(X::DataFrame) = colnorm2(ensure_mat(X))

function colnorm2(X::AbstMatVec{Q}) where Q <: Float
    s = zeros(eltype(X), nco(X))
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            s[j] += X[i, j]^2
        end
    end
    s
end

function colnorm2(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float
    s = zeros(eltype(X), nco(X))
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            s[j] += X[i, j]^2 * weights.values[i]
        end
    end
    s
end

"""
    colvar(X::DataFrame)
    colvar(X::AbstMatVec{Q}) where Q <: Float
    colvar(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float 
Column-wise (uncorrected) variances of a matrix.
* `X` : Matrix (n, p) or vector (n).
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
colvar(X::DataFrame) = colvar(ensure_mat(X))

function colvar(X::AbstMatVec{Q}) where Q <: Float
    s = similar(X, nco(X))
    Threads.@threads for j in axes(X, 2)
        s[j] = varv(vcol(X, j))
    end
    s
end

function colvar(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float
    s = similar(X, nco(X))
    Threads.@threads for j in axes(X, 2)
        s[j] = varv(vcol(X, j), weights)
    end
    s
end

"""
    colstd(X::DataFrame)
    colstd(X::AbstMatVec{Q}) where Q <: Float
    colstd(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float
Column-wise (uncorrected) standard deviations of a matrix.
* `X` : Matrix (n, p) or vector (n).
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
colstd(X::DataFrame) = colstd(ensure_mat(X))

colstd(X::AbstMatVec{Q}) where Q <: Float = sqrt.(colvar(X))

colstd(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float = sqrt.(colvar(X, weights))

"""
    colprt(X::DataFrame)
    colprt(X::AbstMatVec{Q}) where Q <: Float
    colprt(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float
Column-wise (uncorrected) standard deviations of a matrix.
* `X` : Matrix (n, p) or vector (n).
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
colprt(X::DataFrame) = colprt(ensure_mat(X))

colprt(X::AbstMatVec{Q}) where Q <: Float = sqrt.(colstd(X))

colprt(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float = sqrt.(colstd(X, weights))

"""
    colmed(X::DataFrame)
    colmed(X::AbstMatVec{Q}) where Q <: Float
Column-wise medians of a matrix.
* `X` : Matrix (n, p) or vector (n).

Return a vector (p).

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)

colmed(X)
```
""" 
colmed(X::DataFrame) = colmed(ensure_mat(X))

function colmed(X::AbstMatVec{Q}) where Q <: Float
    s = similar(X, nco(X))
    Threads.@threads for j in axes(X, 2)
        s[j] = Statistics.median(vcol(X, j))
    end
    s
end

"""
    colmad(X::DataFrame)
    colmad(X::AbstMatVec{Q}) where Q <: Float 
Column-wise median absolute deviations (MAD) of a matrix.
* `X` : Matrix (n, p) or vector (n).

Return a vector (p).

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)

colmad(X)
```
"""
colmad(X::DataFrame) = colmad(ensure_mat(X))

function colmad(X::AbstMatVec{Q}) where Q <: Float
    s = similar(X, nco(X))
    Threads.@threads for j in axes(X, 2)
        s[j] = madv(vcol(X, j))
    end
    s
end

colmad(X::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where Q <: Float = colmad(X)  # for consistency when weights

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

function colsumskip(X::AbstractArray{Union{Missing, Q}}, 
        weights::ProbabilityWeights{Q}) where Q <: Float
    X = ensure_mat(X)
    v = zeros(Q, nco(X))
    @inbounds for j in axes(X, 2)
        s = ismissing.(vcol(X, j))
        w = pweight(rmrow(weights.values, s))
        v[j] = sum(w.values .* rmrow(X[:, j], s))
    end
    v
end

##
colmeanskip(X) = [Statistics.mean(skipmissing(x)) for x in eachcol(ensure_mat(X))]

colmeanskip(X::AbstractArray{Union{Missing, Q}}, 
        weights::ProbabilityWeights{Q}) where Q <: Float = colsumskip(X, weights)

##
colstdskip(X) = [Statistics.std(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]

colstdskip(X::AbstractArray{Union{Missing, Q}}, 
        weights::ProbabilityWeights{Q}) where Q <: Float = sqrt.(colvarskip(X, weights))

##
colvarskip(X) = [Statistics.var(skipmissing(x); corrected = false) for x in eachcol(ensure_mat(X))]

function colvarskip(X::AbstractArray{Union{Missing, Q}}, 
        weights::ProbabilityWeights{Q}) where Q <: Float
    p = nco(X)
    v = colmeanskip(X, weights)
    @inbounds for j in axes(X, 2)
        s = ismissing.(vcol(X, j))
        w = pweight(rmrow(weights.values, s))
        v[j] = dot(w.values, (rmrow(X[:, j], s) .- v[j]).^2)        
    end
    v 
end



