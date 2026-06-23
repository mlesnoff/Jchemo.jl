"""
    tab(X::AbstractArray)
    tab(datf::DataFrame; group = nothing)
Tabulation of categorical variables.
* `X` : A categorical array or variable to tabulate.
* `datf` : A dataframe containing categorical variable(s) to tabulate.
Specific for a dataset:
* `group` : Vector of the names (String or Symbol) of the group variables to consider in dataframe `datf` 
    (by default, all the columns of `datf`).

The function returns sorted levels. It does not support inputs of type `Any`.

## Examples
```julia
using Jchemo, DataFrames

x = rand(1:3, 20)

res = tab(x)
res.keys
res.vals

tab(string.(x))

n = 20
X = hcat(rand(["1"; "2"], n), rand(["a", "b", "c"], n))
datf = DataFrame(X, [:v1, :v2])

tab(X[:, 2])
tab(X)

tab(datf)
tab(datf; group = [:v2])
tab(datf; group = [:v1, :v2])
```
"""
tab(X::AbstractArray) = sort(StatsBase.countmap(X))

function tab(datf::DataFrame; group = nothing)
    dat = copy(datf)
    if isnothing(group) ; group = names(dat) ; end
    dat.n = ones(Int, nro(dat))
    Q = eltype(group)
    res = aggstat(dat; sel = [Q(:n)], group, algo = sum)
    res
end

"""
    tabdupl(x)
Tabulate duplicated values in a vector.
* `x` : Categorical variable.

## Examples
```julia
using Jchemo

x = ["a", "b", "c", "a", "b", "b"]
tab(x)
res = tabdupl(x)
res.keys
res.vals
```
"""
function tabdupl(x)
    z = tab(x)
    s = z.vals .> 1
    u = z.keys[s]
    tab(x[in(u).(x)])
end

"""
    tabcont(x::Vector{Q}, q::Vector{Q}) where Q <: AbstractFloat
Tabulate a continuous variable.
* `x` : Continuous variable (n).
* `q` : Numerical values (K) separating the class levels from `x`.  

The function returns K + 1 levels. For a given value x of vector `x` and `q` a vector 
of length K: 
* x <= q[1]             : ==> 1 
* q[1] < x <= q[2]      : ==> 2 
* etc.
* q[K - 1] < x <= q[K]  : ==> K 
* q[K] < x              : ==> K + 1 

## Examples
```julia
using Jchemo

x = rand(100)
q = [.01; .5; .500001; .9; 1.1]

res = tabcont(x, q)
sum(res.n)
```
"""
function tabcont(x::Vector{Q}, q::Vector{Q}) where Q <: AbstractFloat
    bin = mbin(q)
    nbin = length(bin)
    lev = collect(1:nbin)
    v = recod_contbylev(x, q)
    resv = tab(v)
    val = zeros(Int, nbin)
    for i in eachindex(lev) 
        k = findfirst(lev[i] .== resv.keys)
        if !isnothing(k)
            val[i] = resv.vals[k]
        end
    end
    val
    DataFrame(bin = bin, lev = lev, n = val)
end

"""
    mbin(q::Vector{Q}) where Q <: AbstractFloat
Build histogram-bin intervals.
* `q` : Numerical values (K) defining the limits of the intervals. 

For a given vector `q` of length K, the function returns K + 1 intervals: 
* (-Inf, q[1]]
* (q[1], q[2]]
* etc.
* (q[K - 1], q[K]]
* (q[K], Inf)

## Examples
```julia
using Jchemo

q = [.01; .5; .500001; .9; 1.1]
mbin(q)
```
"""
function mbin(q::Vector{Q}) where Q <: AbstractFloat
    zq = vcat(-Inf, q, Inf)
    bin = list(Vector{Q}, length(q) + 1)
    for i in eachindex(bin)
        bin[i] = [zq[i]; zq[i + 1]]
    end
    bin
end

