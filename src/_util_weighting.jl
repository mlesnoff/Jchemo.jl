###### Building weights

""" 
    pweight(x::Vector{Q}) where Q <: Real
    pweight(T::DataType, x::Vector{Q}) where {Q <: Real}
Wrapper of function `StatsBase.pweights` returning an object of type `StatsBase.ProbabilityWeights`.

The wrapper forces the probability weights to sum to 1:
* The returned object `values` is equal to `x / sum(x)`.

## Examples
```julia
using Jchemo

x = rand(5)
weights = pweight(x)
@names weights 
weights.values 
sum(weights.values) 
weights.sum
```
"""
pweight(x::Vector{Q}) where Q <: Real = StatsBase.pweights(x / sum(x))
#function pweight(x)
#    tot = sum(x) 
#    ProbabilityWeights(x / tot, one(eltype(x)))
#end

pweight(T::DataType, x::Vector{Q}) where Q <: Real = pweight(T.(x))

""" 
    pweightcla(y::Vector{String}; 
        prior::Union{Symbol, Vector{Q}} = :prop) where Q <: Float
    pweightcla(T::DataType, y::Vector{String}; 
        prior::Union{Symbol, Vector{Q}} = :prop) where Q <: Float
Compute observation weights for a categorical variable, given specified sub-total weights for the classes.
* `y` : A categorical variable (class membership) (n). Must be a `Vector{String}`.
* `Q` : A data type (e.g., `Float32`).
Keyword arguments:
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).

Return an object of type `ProbabilityWeights` (see function `pweight`) containing a vector `weights.values` (n) 
that sums to 1.

## Examples
```julia
using Jchemo

y = vcat(rand(["a" ; "c"], 900), fill("b", 100))
tab(y)
weights = pweightcla(y)
#weights = pweightcla(y; prior = :prop)
#weights = pweightcla(y; prior = :unif)
#weights = pweightcla(y; prior = [.1, .7, .2])
res = aggstat(weights.values, y; algo = sum)
[res.lev res.X]
```
"""
pweightcla(y::Vector{String}; 
    prior::Union{Symbol, Vector{Q}} = :prop) where Q <: Float = pweightcla(Float64, y; prior) 

function pweightcla(T::DataType, y::Vector{String}; 
        prior::Union{Symbol, Vector{Q}} = :prop) where Q <: Float
    n = length(y)
    res = tab(y)
    lev = res.keys
    nlev = length(lev)
    vals = T.(res.vals)
    if isequal(prior, :unif)
        priors = ones(T, nlev) / nlev
    elseif isequal(prior, :prop)
        priors = vals / n
    else
        priors = pweight(T, prior).values  # could be '= prior', but pweight not costly 
    end
    w = zeros(T, n)
    @inbounds for i in eachindex(lev)
        s = y .== lev[i]
        w[s] .= priors[i] / res.vals[i]
    end
    pweight(w)
end

##### Weighting rows or columns

"""
    fweightr(X::AbstMatVec{Q}, v::Vector{Q}) where Q <: Float
    fweightr!(X::AbstMatVec{Q}, v::Vector{Q}) where Q <: Float
Weight each row of a matrix.
* `X` : Matrix (n, p) or vector (n).
* `v` : A weighting vector (n).

## Examples
```julia
using Jchemo, LinearAlgebra

X = rand(5, 2) 
v = rand(5) 
fweightr(X, v)
diagm(v) * X

fweightr!(X, v)
X
```
""" 
fweightr(X::AbstMatVec{Q}, v::Vector{Q}) where Q <: Float = v .* X

fweightr!(X::AbstMatVec{Q}, v::Vector{Q}) where Q <: Float = X .= v .* X

"""
    fweightc(X::AbstractMatrix{Q}, v::Vector{Q}) where Q <: Float
    fweightc!(X::AbstractMatrix{Q}, v::Vector{Q}) where Q <: Float
Weight each column of a matrix.
* `X` : Matrix (n, p) or vector (n).
* `v` : A weighting vector (p).

## Examples
```julia
using Jchemo, LinearAlgebra

X = rand(5, 2) 
v = rand(2) 
fweightc(X, v)
X * diagm(v)

fweightc!(X, v)
X
```
""" 
fweightc(X::AbstractMatrix{Q}, v::Vector{Q}) where Q <: Float = v' .* X

fweightc!(X::AbstractMatrix{Q}, v::Vector{Q}) where Q <: Float = X .= v' .* X


