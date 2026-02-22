###### Building weights

""" 
    pweight(x::Vector)
Wrapper of function `StatsBase.pweights` returning an object of type `StatsBase.ProbabilityWeights`.

The wrapper forces the probability weights to sum to 1:
* The returned object `values` is equal to `x / sum(x)`.

## Examples
```julia
using Jchemo

x = rand(10)
w = pweight(x)
@names w 
w.values 
sum(w.values) 
w.sum
```
"""
pweight(x) = pweights(x / sum(x))
#function pweight(x)
#    tot = sum(x) 
#    ProbabilityWeights(x / tot, one(eltype(x)))
#end

""" 
    pweightcla(y::AbstractVector; prior::Union{Symbol, Vector} = :prop)
    pweightcla(Q::DataType, y::Vector; prior::Union{Symbol, Vector} = :prop)
Compute observation weights for a categorical variable, given specified sub-total weights for the classes.
* `y` : A categorical variable (n) (class membership).
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

y = vcat(rand(["a" ; "c"], 900), repeat(["b"], 100))
tab(y)
weights = pweightcla(y)
#weights = pweightcla(y; prior = :prop)
#weights = pweightcla(y; prior = [.1, .7, .2])
res = aggstat(weights.values, y; algo = sum)
[res.lev res.X]
```
"""
function pweightcla(y::AbstractVector; prior::Union{Symbol, Vector} = :prop)
    n = length(y)
    res = tab(y)
    lev = res.keys
    nlev = length(lev)
    if isequal(prior, :unif)
        priors = ones(nlev) / nlev
    elseif isequal(prior, :prop)
        priors = res.vals / n
    else
        priors = pweight(prior).values  # could be '= prior', but pweight not costly 
    end
    w = zeros(n)
    @inbounds for i in eachindex(lev)
        s = y .== lev[i]
        w[s] .= priors[i] / res.vals[i]
    end
    pweight(w)
end

function pweightcla(Q::DataType, y::AbstractVector; prior::Union{Symbol, Vector} = :prop)
    pweight(convert.(Q, pweightcla(y; prior).values))
end

##### Weighting entire rows or columns

"""
    rweight(X, v)
    rweight!(X::AbstractMatrix, v)
Weight each row of a matrix.
* `X` : Data (n, p).
* `v` : A weighting vector (n).

## Examples
```julia
using Jchemo, LinearAlgebra

X = rand(5, 2) 
w = rand(5) 
rweight(X, w)
diagm(w) * X

rweight!(X, w)
X
```
""" 
function rweight(X, v)
    X = ensure_mat(X)
    n, p = size(X)
    zX = similar(v, n, p)
    @inbounds for j = 1:p, i = 1:n
        zX[i, j] = X[i, j] * v[i]
    end  
    zX
end

function rweight!(X::AbstractMatrix, v)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] *= v[i]
    end
end

"""
    cweight(X, v)
    cweight!(X::AbstractMatrix, v)
Weight each column of a matrix.
* `X` : Data (n, p).
* `v` : A weighting vector (p).

## Examples
```julia
using Jchemo, LinearAlgebra

X = rand(5, 2) 
w = rand(2) 
cweight(X, w)
X * diagm(w)

cweight!(X, w)
X
```
""" 
function cweight(X, v)
    X = ensure_mat(X)
    n, p = size(X)
    zX = similar(v, n, p)
    @inbounds for j = 1:p, i = 1:n
        zX[i, j] = X[i, j] * v[j]
    end  
    zX
end

function cweight!(X::AbstractMatrix, v)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] *= v[j]
    end
end

