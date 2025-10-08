"""
    aggmean(X, y)
Compute column-wise mean by class in a dataset.
* `X` : Data (n, p).
* `y` : A categorical variable (n) (class membership).

Faster than `aggstat`. 

## Examples
```julia
using Jchemo

n, p = 20, 5
X = rand(n, p)
y = rand(1:3, n)
df = DataFrame(X, :auto) 
res = aggmean(X, y)
res.X
res.lev 
aggmean(df, y).X
```
""" 
function aggmean(X, y) 
    X = ensure_mat(X)
    y = vec(y)
    p = nco(X)
    lev = mlev(y)
    nlev = length(lev)
    zX = similar(X, nlev, p)
    @inbounds for i in eachindex(lev)
    #Threads.@threads for i in eachindex(lev)
        zX[i, :] .= colmean(vrow(X, y .== lev[i]))
    end
    (X = zX, lev)
end

"""
    aggstat(X, y; algo = mean)
    aggstat(X::DataFrame; sel, groupby, algo = mean)
Compute column-wise statistics by group in a dataset.
* `X` : Data (n, p).
* `y` : A categorical variable (n) defining the groups.
* `algo` : Function to compute (default = mean).
Specific for `X::dataframe`:
* `sel` : Names (vector) of the variables to summarize.
* `groupby` : Names (vector) of the categorical variables defining the groups.

Variables defined in `sel` and `groupby` must be columns of `X`.

## Examples
```julia
using Jchemo, DataFrames, Statistics

n, p = 20, 5
X = rand(n, p)
df = DataFrame(X, :auto)
y = rand(1:3, n)
res = aggstat(X, y; algo = sum)
@names res
res.lev 
res.X
aggstat(df, y; algo = sum).X

n, p = 20, 5
X = rand(n, p)
df = DataFrame(X, string.("v", 1:p))
df.y1 = rand(1:2, n)
df.y2 = rand(["a", "b", "c"], n)
df
aggstat(df; sel = [:v1, :v2] , groupby = [:y1, :y2], algo = var)  # return a dataframe 
```
""" 
function aggstat(X, y; algo = mean)
    X = ensure_mat(X)
    y = vec(y)
    n, p = size(X)
    lev = mlev(y)
    nlev = length(lev)
    zX = similar(X, nlev, p)
    s = BitVector(list(Bool, n))
    @inbounds for i in eachindex(lev), j = 1:p
        s .= y .== lev[i]
        zX[i, j] = algo(view(X, s, j))
    end
    (X = zX, lev)
end

function aggstat(X::DataFrame; sel, groupby, algo = mean)
    gdf = groupby(X, groupby) 
    res = combine(gdf, sel .=> algo, renamecols = false)
    sort!(res, groupby)
end

""" 
    aggsumv(x::Vector, y::Union{Vector, BitVector})
Compute sub-total sums by class of a categorical variable.
* `x` : A quantitative variable to sum (n) 
* `y` : A categorical variable (n) (class membership).

## Examples
```julia
using Jchemo

x = rand(1000)
y = vcat(rand(["a" ; "c"], 900), repeat(["b"], 100))
aggsumv(x, y)
```
"""
function aggsumv(x::Vector, y::Union{Vector, BitVector})
    lev = mlev(y)
    v = similar(x, length(lev)) 
    @inbounds for i in eachindex(lev) 
        s = y .== lev[i]
        v[i] = sum(vrow(x, s))
    end
    v
end

""" 
    convertdf(df::DataFrame; miss = nothing, typ)
Convert the columns of a dataframe to given types.
* `df` : A dataframe.
* `miss` : The code used in `df` to identify the data to be declared as `missing` (of type `Missing`).
    See function `recod_miss`.
* `typ` : A vector of the targeted types for the columns of the new dataframe.  

## Examples
```julia
using Jchemo, DataFrames
```
"""
function convertdf(df::DataFrame; miss = nothing, typ)
    df = string.(df)
    df = recod_miss(df; miss = string(miss))
    res = DataFrame()
    for i in eachindex(typ)
        z = df[:, i]
        if typ[i] == String
            sum(ismissing.(z)) == 0 ? z = string.(z) : nothing
        else
            if sum(ismissing.(z)) == 0
                z = parse.(typ[i], z)
            else
                z = parsemiss(typ[i], z)
            end
        end
        res = hcat(res, z; makeunique = true)
    end
    rename!(res, names(df))
    res
end

"""
    dummy(y)
Compute dummy table from a categorical variable.
* `y` : A categorical variable.

The output `Y` (dummy table) is a BitMatrix.

## Examples
```julia
using Jchemo

y = ["d", "a", "b", "c", "b", "c"]
#y =  rand(1:3, 7)
res = dummy(y)
@names res
res.Y
```
"""
function dummy(y)
    lev = mlev(y)
    ## Thanks to the idea given in this post (@Mattriks):
    ## https://discourse.julialang.org/t/all-the-ways-to-do-one-hot-encoding/64807/4
    Y = y .== permutedims(lev)
    (Y = Y, lev)
end

"""
    dupl(X; digits = 3)
Find duplicated rows in a dataset.
* `X` : A dataset.
* `digits` : Nb. digits used to round `X` before checking.

## Examples
```julia
using Jchemo

X = rand(5, 3)
Z = vcat(X, X[1:3, :], X[1:1, :])
dupl(X)
dupl(Z)

M = hcat(X, fill(missing, 5))
Z = vcat(M, M[1:3, :])
dupl(M)
dupl(Z)
```
"""
function dupl(X; digits = 3)
    X = ensure_mat(X)
    ## round, etc. does not accept missing values
    X[ismissing.(X)] .= -1e5
    ## End
    X = round.(X, digits = digits)
    n = nro(X)
    rownum1 = []
    rownum2 = []
    @inbounds for i = 1:n
        @inbounds for j = (i + 1):n
            res = isequal(vrow(X, i), vrow(X, j))
            if res
                push!(rownum1, i)
                push!(rownum2, j)
            end
        end
    end
    u = findall(rownum1 .!= rownum2)
    res = DataFrame((rownum1 = rownum1[u], rownum2 = rownum2[u]))
    Int.(res)
end

"""
    ensure_df(X)
Reshape `X` to a dataframe if necessary.
"""
ensure_df(X::DataFrame) = X
ensure_df(X::AbstractVector) = DataFrame([X], :auto)
ensure_df(X::AbstractMatrix) = DataFrame(X, :auto)

"""
    ensure_mat(X)
Reshape `X` to a matrix if necessary.
"""
ensure_mat(X::AbstractMatrix) = X
## Tentative to allow the use of CUDA
## Old was: ensure_mat(X::AbstractVector) = Matrix(reshape(X, :, 1))
ensure_mat(X::AbstractVector) = reshape(X, :, 1)
## End
ensure_mat(X::Number) = reshape([X], 1, 1)
ensure_mat(X::LinearAlgebra.Adjoint) = Matrix(X)
ensure_mat(X::DataFrame) = Matrix(X)

"""
    findmax_cla(x)
    findmax_cla(x, weights::Weight)
Find the most occurent level in `x`.
* `x` : A categorical variable.
* `weights` : Weights (n) of the observations. Object of type `Weight` (e.g. generated by function `mweight`).

If ex-aequos, the function returns the first.

## Examples
```julia
using Jchemo

x = rand(1:3, 10)
tab(x)
findmax_cla(x)
```
"""
function findmax_cla(x)
    n = length(x)
    res = aggstat(ones(n), x; algo = sum)
    res.lev[argmax(res.X)]   # if equal, argmax takes the first
end

function findmax_cla(x, weights::Weight)
    res = aggstat(weights.w, x; algo = sum)
    res.lev[argmax(res.X)]   
end

"""
    findmiss(X)
Find rows with missing data in a dataset.
* `X` : A dataset.

For dataframes, see also `DataFrames.completecases` and `DataFrames.dropmissing`.

## Examples
```julia
using Jchemo

X = rand(5, 4)
zX = hcat(rand(2, 3), fill(missing, 2))
Z = vcat(X, zX)
findmiss(X)
findmiss(Z)
```
"""
function findmiss(X)
    X = ensure_mat(X)
    z = vec(sum(ismissing.(X); dims = 2))
    u = findall(z .> 0) 
    DataFrame(:rownum => u, :nmissing => z[u])
end

"""
    finduniq(id)
Find the indexes making unique the IDs in a ID vector.
* `id` : A vector of IDs.

Can be used to remove duplicated rows in a dataset, identified by a single ID variable.

## Examples
```julia
using Jchemo

v = ["a", "d", "c", "b", "a", "d", "a"]  # a vector of IDs

s = finduniq(v)  # indexes of the IDs without duplicates
v[s]  
```
"""
function finduniq(id)
    n = length(id)
    res = tabdupl(id)
    idd = res.keys
    s = list(Int, 0) 
    for i in eachindex(idd)
        zs = findall(id .== idd[i])
        append!(s, zs[2:end])
    end
    rmrow(collect(1:n), s)
end

"""
    @head X
Display the first rows of a dataset.

## Examples
```julia
using Jchemo

X = rand(100, 5)
@head X
```
"""
function head(X)
    n = nro(X) 
    m = min(3, n)
    if isa(X, AbstractVector)
        display(X[1:m])
    else
        display(X[1:m, :])
    end
    if n > 3
        println("... ", size(X))
    end
    println(" ")
end

macro head(X)
    esc( :( Jchemo.head($X) ))
end

"""
    list(n::Integer)
Create a Vector{Any}(nothing, n).

`isnothing(object, i)` can be used to check if cell i is empty.

## Examples
```julia
using Jchemo

list(5)
```
"""  
list(n::Integer) = Vector{Any}(nothing, n) 

"""
    list(Q, n::Integer)
Create a Vector `{Q}(undef, n)`.

`isassigned(object, i)` can be used to check if cell i is empty.

## Examples
```julia
using Jchemo

list(Float64, 5)
list(Array{Float64}, 5)
list(Matrix{Int}, 5)
```
"""  
list(Q, n::Integer) = Vector{Q}(undef, n)

""" 
    mlev(x)
Return the sorted levels of a vector or a dataset. 

## Examples
```julia
using Jchemo

x = rand(["a";"b";"c"], 20)
lev = mlev(x)
nlev = length(lev)

X = reshape(x, 5, 4)
mlev(X)

df = DataFrame(g1 = rand(1:2, n), g2 = rand(["a"; "c"], n))
mlev(df)
```
"""
mlev(x) = sort(unique(x)) 

""" 
    mweight(x::Vector)
Return an object of type `Weight` containing vector `w = x / sum(x)` (if ad'hoc building, `w` must sum to 1).

## Examples
```julia
using Jchemo

x = rand(10)
w = mweight(x)
sum(w.w)
```
"""
mweight(x::Vector) = Weight(x / sum(x))

#mweight(x::AbstractVector) = Weight(x / sum(x))  # For CUDA

#mweight(w::Vector{Q}) where {Q <: AbstractFloat} = mweight!(copy(w))
#mweight!(w::Vector{Q}) where {Q <: AbstractFloat} = w ./= sum(w)

#mweight(w::Union{Vector{Float32}, Vector{Float64}}) = mweight!(copy(w))
#mweight!(w::Union{Vector{Float32}, Vector{Float64}}) = w ./= sum(w)

""" 
    mweightcla(y::AbstractVector; prior::Union{Symbol, Vector} = :prop)
    mweightcla(Q::DataType, y::Vector; prior::Union{Symbol, Vector} = :prop)
Compute observation weights for a categorical variable, given specified sub-total weights for the classes.
* `y` : A categorical variable (n) (class membership).
* `Q` : A data type (e.g. `Float32`).
Keyword arguments:
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).

Return an object of type `Weight` (see function `mweight`) containing a vector `w` (n) that sums to 1.

## Examples
```julia
using Jchemo

y = vcat(rand(["a" ; "c"], 900), repeat(["b"], 100))
tab(y)
weights = mweightcla(y)
#weights = mweightcla(y; prior = :prop)
#weights = mweightcla(y; prior = [.1, .7, .2])
res = aggstat(weights.w, y; algo = sum)
[res.lev res.X]
```
"""
function mweightcla(y::AbstractVector; prior::Union{Symbol, Vector} = :prop)
    n = length(y)
    res = tab(y)
    lev = res.keys
    nlev = length(lev)
    if isequal(prior, :unif)
        priors = ones(nlev) / nlev
    elseif isequal(prior, :prop)
        priors = res.vals / n
    else
        priors = mweight(prior).w  # could be '= prior', but mweight not costly 
    end
    w = zeros(n)
    @inbounds for i in eachindex(lev)
        s = y .== lev[i]
        w[s] .= priors[i] / res.vals[i]
    end
    mweight(w)
end

function mweightcla(Q::DataType, y::AbstractVector; prior::Union{Symbol, Vector} = :prop)
    mweight(convert.(Q, mweightcla(y; prior).w))
end

"""
    @namvar x
Return the name of a variable.
* `x` : A variable or function.

Thanks to: 
https://stackoverflow.com/questions/38986764/save-variable-name-as-string-in-julia

## Examples
```julia
using Jchemo

z = 1:5
Jchemo.@namvar z
```
"""
macro namvar(arg)
    x = string(arg)
    quote
        $x
    end
end


""" 
    nco(X)
Return the nb. columns of `X`.
"""
nco(X) = size(X, 2)

""" 
    nro(X)
Return the nb. rows of `X`.
"""
nro(X) = size(X, 1)

""" 
    out(x)
Return if elements of a vector are strictly outside of a given range.
* `x` : Univariate data.
* `y` : Univariate data on which is computed the range (min, max).

Return a BitVector.

## Examples
```julia
using Jchemo

x = [-200.; -100; -1; 0; 1; 200]
out(x, [-1; .2; 1])
out(x, (-1, 1))
```
"""
out(x, y) = (x .< minimum(y)) .| (x .> maximum(y))

""" 
    parsemiss(Q, x::Vector{Union{String, Missing}})
Parsing a string vector allowing missing data.
* `Q` : Type that results from the parsing of type `String'. 
* `x` : A string vector containing `missing` (of type `Missing`) observations.

See examples.

## Examples
```julia
using Jchemo

x = ["1"; "3.2"; missing]
x_p = parsemiss(Float64, x)
```
"""
function parsemiss(Q, x::Vector{Union{String, Missing}})
    v = missings(Q, length(x))
    for i in eachindex(x)
        ismissing(x[i]) ? nothing : v[i] = parse(Q, x[i])
    end
    v
end

"""
    pval(d::Distribution, q)
    pval(x::Array, q)
    pval(e_cdf::ECDF, q)
Compute p-value(s) for a distribution, an ECDF or vector.
* `d` : A distribution computed from `Distribution.jl`.
* `x` : Univariate data.
* `e_cdf` : An ECDF computed from `StatsBase.jl`.
* `q` : Value(s) for which to compute the p-value(s).

Compute or estimate the p-value of quantile `q`, ie. V(Q > `q`) where Q is the random variable.

## Examples
```julia
using Jchemo, Distributions, StatsBase

d = Distributions.Normal(0, 1)
q = 1.96
#q = [1.64; 1.96]
Distributions.cdf(d, q)    # cumulative density function (CDF)
Distributions.ccdf(d, q)   # complementary CDF (CCDF)
pval(d, q)                 # Distributions.ccdf

x = rand(5)
e_cdf = StatsBase.ecdf(x)
e_cdf(x)                # empirical CDF computed at each point of x (ECDF)
p_val = 1 .- e_cdf(x)   # complementary ECDF at each point of x
q = .3
#q = [.3; .5; 10]
pval(e_cdf, q)          # 1 .- e_cdf(q)
pval(x, q)
```
"""
pval(d::Distribution, q) = Distributions.ccdf(d, q)

pval(e_cdf::ECDF, q) = 1 .- e_cdf(q)

pval(x::AbstractVector, q) = pval(StatsBase.ecdf(x), q)

"""
    recovkw(ParStruct, kwargs)
"""
function recovkw(ParStruct::DataType, kwargs)
    if length(Dict(kwargs)) == 0
        kwargs_new = kwargs
        par = ParStruct()
    else
        args = sort(collect(kwargs), by = x -> x[1])
        z1 = fieldnames(ParStruct)
        z2 = sort(collect(keys(Dict(args))))
        s = in(z1).(z2)
        if sum(s) > 0
            kwargs_new = args[s]
            par = [ParStruct(; Dict(kws)...) for kws in zip([[k => v] for (k, v) in kwargs_new]...)][1]
        else
            kwargs_new = nothing
            par = ParStruct()
        end
    end
    (kwargs = kwargs_new, par)
end

recovkw(ParStruct::DataType) = (kwargs = nothing, par = ParStruct())

"""
    rmcol(X::Union{AbstractMatrix, DataFrame}, s::Union{Vector, BitVector, UnitRange, Number})
    rmcol(X::Vector, s::Union{Vector, BitVector, UnitRange, Number})
Remove the columns of a matrix or the components of a vector 
having indexes `s`.
* `X` : Matrix or vector.
* `s` : Vector of the indexes.

## Examples
```julia
using Jchemo

X = rand(5, 3) 
rmcol(X, [1, 3])
```
"""
function rmcol(X::Union{AbstractMatrix, DataFrame}, s::Union{Vector, BitVector, UnitRange, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[:, setdiff(1:end, Int.(s))]
end

function rmcol(X::Vector, s::Union{Vector, BitVector, UnitRange, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[setdiff(1:end, Int.(s))]
end

"""
    rmrow(X::Union{AbstractMatrix, DataFrame}, s::Union{Vector, BitVector, UnitRange, Number})
    rmrow(X::Union{Vector, BitVector}, s::Union{Vector, BitVector, UnitRange, Number})
Remove the rows of a matrix or the components of a vector having indexes `s`.
* `X` : Matrix or vector.
* `s` : Vector of the indexes.

## Examples
```julia
using Jchemo

X = rand(5, 2) 
rmrow(X, [1, 4])
```
"""
function rmrow(X::Union{AbstractMatrix, DataFrame}, s::Union{Vector, BitVector, UnitRange, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[setdiff(1:end, Int.(s)), :]
end

function rmrow(X::Union{Vector, BitVector}, s::Union{Vector, BitVector, UnitRange, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[setdiff(1:end, Int.(s))]
end

"""
    softmax(x::AbstractVector)
    softmax(X::Union{Matrix, DataFrame})
Softmax function.
* `x` : A vector to transform.
* `X` : A matrix whose rows are transformed.

Let v be a vector:
* 'softmax'(v) = exp.(v) / sum(exp.(v))

## Examples
```julia
using Jchemo

x = 1:3
softmax(x)

X = rand(5, 3)
softmax(X)
```
"""
function softmax(x::AbstractVector)
    expx = exp.(x) 
    expx / sum(expx)
end

function softmax(X::Union{Matrix, DataFrame})
    X = ensure_mat(X)
    n = nro(V)
    V = similar(X)
    @inbounds for i = 1:n
        V[i, :] .= softmax(vrow(X, i))
    end
    V
end

"""
    sourcedir(path)
Include all the files contained in a directory.
"""
function sourcedir(path)
    z = readdir(path)  ## List of files in path
    for i in eachindex(z)
        include(string(path, "/", z[i]))
    end
end

"""
    summ(X; digits = 3)
    summ(X, y; digits = 3)
Summarize a dataset (or a variable).
* `X` : A dataset (n, p).
* `y` : A categorical variable (n) (class membership).
* `digits` : Nb. digits in the outputs.

## Examples
```julia
using Jchemo

n = 50
X = rand(n, 3) 
y = rand(1:3, n)
res = summ(X)
@names res
summ(X[:, 2]).res

summ(X, y)
```
"""
function summ(X; digits = 3)
    X = ensure_df(X)
    res = StatsBase.describe(X, :mean, :std, :min, :max, :nmissing) 
    insertcols!(res, 6, :n => nro(X) .- res.nmissing)
    for i = 2:4
        z = vcol(res, i)
        s = findall(isa.(z, Real))
        res[s, i] .= round.(res[s, i], digits = digits)
    end
    (res = res, ntot = nro(X))
end

function summ(X, y; digits = 3)
    y = vec(y)
    lev = mlev(y)
    for i in eachindex(lev)
        s = y .== lev[i]
        res = summ(X[s, :]; digits = digits).res
        println("Class: ", lev[i])
        println(res)
        println("") ; println("") 
    end
end

"""
    tab(X::AbstractArray)
    tab(X::DataFrame; groupby = nothing)
Tabulation of categorical variables.
* `x` : Categorical variable or dataset containing categorical variable(s).
Specific for a dataset:
* `groupby` : Vector of the names of the group variables to consider in `X` (by default: all the columns of `X`).

The output cointains sorted levels.

## Examples
```julia
using Jchemo, DataFrames

x = rand(["a"; "b"; "c"], 20)
res = tab(x)
res.keys
res.vals

n = 20
X = hcat(rand(1:2, n), rand(["a", "b", "c"], n))
df = DataFrame(X, [:v1, :v2])

tab(X[:, 2])
tab(string.(X))

tab(df)
tab(df; groupby = [:v1, :v2])
tab(df; groupby = :v2)
```
"""
tab(X::AbstractArray) = sort(StatsBase.countmap(vec(X)))

function tab(X::DataFrame; groupby = nothing)
    zX = copy(X)
    isa(zX, Vector) ? zX = DataFrame(x1 = zX) : nothing
    isa(zX, DataFrame) ? nothing : zX = DataFrame(zX, :auto)
    isnothing(groupby) ? groupby = names(zX) : nothing
    zX.n = ones(nro(zX))
    res = aggstat(zX; sel = :n, groupby = groupby, algo = sum)
    res.n = Int.(res.n)
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
    thresh_hard(x::Real, delta)
Hard thresholding function.
* `x` : Value (scalar) to transform.
* `delta` : Range for the thresholding.

The returned value is:
* abs(`x`) > `delta` ? `x` : 0
where delta >= 0.

## Examples
```julia
using Jchemo, CairoMakie 

delta = .7
thresh_hard(3, delta)

x = LinRange(-2, 2, 500)
y = thresh_hard.(x, delta)
lines(x, y; axis = (xlabel = "x", ylabel = "f(x)"))
```
"""
function thresh_hard(x, delta)
    @assert delta >= 0 "delta must be >= 0."
    abs(x) > delta ? x : zero(eltype(x))
end


"""
    thresh_soft(x::Real, delta)
Soft thresholding function.
* `x` : Value (scalar) to transform.
* `delta` : Range for the thresholding.

The returned value is:
* sign(`x`) * max(0, abs(`x`) - `delta`)
where delta >= 0.

## Examples
```julia
using Jchemo, CairoMakie 

delta = .7
thresh_soft(3, delta)

x = LinRange(-2, 2, 100)
y = thresh_soft.(x, delta)
lines(x, y; axis = (xlabel = "x", ylabel = "f(x)"))
```
"""
function thresh_soft(x, delta)
    @assert delta >= 0 "delta must be >= 0."
    ## same as: abs(x) > delta ? sign(x) * (abs(x) - delta) : zero(eltype(x))
    sign(x) * max(0, abs(x) - delta)  # type consistent
end

"""
    vcatdf(dat; cols = :intersect) 
Vertical concatenation of a list of dataframes.
* `dat` : List (vector) of dataframes.
* `cols` : Determines the columns of the returned dataframe. See ?DataFrames.vcat.

## Examples
```julia
using Jchemo, DataFrames

dat1 = DataFrame(rand(5, 2), [:v3, :v1]) 
dat2 = DataFrame(100 * rand(2, 2), [:v3, :v1])
dat = (dat1, dat2)
Jchemo.vcatdf(dat)

dat2 = DataFrame(100 * rand(2, 2), [:v1, :v3])
dat = (dat1, dat2)
Jchemo.vcatdf(dat)

dat2 = DataFrame(100 * rand(2, 3), [:v3, :v1, :a])
dat = (dat1, dat2)
Jchemo.vcatdf(dat)
Jchemo.vcatdf(dat; cols = :union)
```
""" 
function vcatdf(dat; cols = :intersect) 
    n = length(dat) 
    X = copy(dat[1])
    group = repeat([1], nro(X))
    if n > 1
        for i = 2:n
            X = DataFrames.vcat(X, dat[i]; cols = cols)
            group = vcat(group, repeat([i], nro(dat[i])))
        end
    end
    (X = X, group)
end

"""
    vcol(X::AbstractMatrix, j)
    vcol(X::DataFrame, j)
    vcol(x::Vector, j)
View of the j-th column(s) of a matrix `X`, or of the j-th element(s) of vector `x`.
""" 
vcol(X, j) = view(X, :, j)
vcol(x::Vector, i) = view(x, i)
vcol(X::DataFrame, j) = view(Matrix(X), :, j)

"""
    vrow(X::AbstractMatrix, i)
    vrow(X::DataFrame, i)
    vrow(x::Vector, i)
View of the i-th row(s) of a matrix `X`, or of the i-th element(s) of vector `x`.
""" 
vrow(X, i) = view(X, i, :) 
vrow(X::DataFrame, i) = view(Matrix(X), i, :)
vrow(x::Vector, i) = view(x, i)

########### Macros 

""" 
    @pmod fun
Shortcut for function `parentmodule`.
* `fun` : The name of a function.

## Examples
```julia
@pmod rand
```
"""
macro pmod(fun)
    esc( :( parentmodule($fun) ))
end

""" 
    @names x
Return the names of the sub-objects contained in a object.
* `x`: An object.
Shortcut for function `propertynames`.
"""
macro names(x)
    esc( :( propertynames($x) ))
end

"""
    @pars fun
Display the keyword arguments (with their default values) of a function.
* `fun` : The name of a function.

## Examples
```julia
using Jchemo

@pars krr
```
"""
macro pars(fun)
    esc( :( Jchemo.defaults($fun) ))
end

""" 
    @plist x
Display each element of a named list.
* `x` : A list.
"""
macro plist(x)
    esc( :( Jchemo.plist($x) ))
end

function plist(x)
    nam = propertynames(x)
    for i in eachindex(nam)
        println("--- ", nam[i])        
        println("")
        println(x[i])
        println("")
    end
end

""" 
    @type x
Display the type and size of a dataset.
* `x` : A dataset.
"""
macro type(x)
    esc( :( Jchemo.ptype($x) ))
end

function ptype(x)
    println(typeof(x))
    println(size(x))
end

