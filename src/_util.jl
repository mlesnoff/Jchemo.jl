"""
    aggstat(X, y; algo = mean)
    aggstat(X::DataFrame; vary, vargroup, algo = mean)
Compute column-wise statistics by class in a dataset.
* `X` : Data (n, p).
* `y` : A categorical variable (n) (class membership).
* `algo` : Function to compute (default = mean).
Specific for dataframes:
* `vary` : Vector of the names of the variables to summarize.
* `vargroup` : Vector of the names of the categorical variables to consider
    for computations by class.

Variables defined in `vary` and `vargroup` must be columns of `X`.

Return a matrix or, if only argument `X::DataFrame` is used, a dataframe.

## Examples
```julia
using Jchemo, DataFrames, Statistics

n, p = 20, 5
X = rand(n, p)
df = DataFrame(X, :auto)
y = rand(1:3, n)
res = aggstat(X, y; algo = sum)
res.X
aggstat(df, y; algo = sum).X

n, p = 20, 5
X = rand(n, p)
df = DataFrame(X, string.("v", 1:p))
df.gr1 = rand(1:2, n)
df.gr2 = rand(["a", "b", "c"], n)
df
aggstat(df; vary = [:v1, :v2], vargroup = [:gr1, :gr2], algo = var)
```
""" 
function aggstat(X, y; algo = mean)
    X = ensure_mat(X)
    y = vec(y)
    q = nco(X)
    lev = mlev(y)
    nlev = length(lev)
    zX = similar(X, nlev, q)
    @inbounds for i in 1:nlev, j = 1:q
        s = y .== lev[i]
        zX[i, j] = algo(view(X, s, j))
    end
    (X = zX, lev)
end

function aggstat(X::DataFrame; vary, vargroup, algo = mean)
    gdf = groupby(X, vargroup) 
    res = combine(gdf, vary .=> algo, renamecols = false)
    sort!(res, vargroup)
end

""" 
    aggsum(x::Vector, y::Union{Vector, BitVector})
Compute sub-total sums by class of a categorical variable.
* `x` : A quantitative variable to sum (n) 
* `y` : A categorical variable (n) (class membership).

Return a vector.

## Examples
```julia
using Jchemo

x = rand(1000)
y = vcat(rand(["a" ; "c"], 900), repeat(["b"], 100))
aggsum(x, y)
```
"""
function aggsum(x::Vector, y::Union{Vector, BitVector})
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
* `miss` : The code used in `df` to identify the data 
    to be declared as `missing` (of type `Missing`).
    See function `recod_miss`.
* `typ` : A vector of the targeted types for the
    columns of the new dataframe.  

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
* `weights` : Weights (n) of the observations. Object of type 
    `Weight` (e.g. generated by function `mweight`).

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
Return an object of type `Weight` containing vector 
`w = x / sum(x)` (if ad'hoc building, `w` must sum to 1).

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
    mweightcla(x::AbstractVector; prior::Union{Symbol, Vector} = :unif)
    mweightcla(Q::DataType, x::Vector; prior::Union{Symbol, Vector} = :unif)
Compute observation weights for a categorical variable, 
    given specified sub-total weights for the classes.
* `x` : A categorical variable (n) (class membership).
* `Q` : A data type (e.g. `Float32`).
Keyword arguments:
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (the vector must be sorted in the same order as `mlev(x)`).

Return an object of type `Weight` (see function `mweight`) containing
a vector `w` (n) that sums to 1.

## Examples
```julia
using Jchemo

x = vcat(rand(["a" ; "c"], 900), repeat(["b"], 100))
tab(x)
weights = mweightcla(x)
#weights = mweightcla(x; prior = :prop)
#weights = mweightcla(x; prior = [.1, .7, .2])
res = aggstat(weights.w, x; algo = sum)
[res.lev res.X]
```
"""
function mweightcla(x::AbstractVector; prior::Union{Symbol, Vector} = :unif)
    n = length(x)
    res = tab(x)
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
        s = x .== lev[i]
        w[s] .= priors[i] / res.vals[i]
    end
    mweight(w)
end

function mweightcla(Q::DataType, x::AbstractVector; prior::Union{Symbol, Vector} = :unif)
    mweight(convert.(Q, mweightcla(x; prior).w))
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
* `x` : A string vector containing `missing` (of type `Missing`) 
    observations.

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

Compute or estimate the p-value of quantile `q`,
ie. V(Q > `q`) where Q is the random variable.

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
    recod_catbydict(x, dict)
 Recode a categorical variable to dictionnary levels.
* `x` : Categorical variable (n) to replace.
* `dict` : Dictionary giving the correpondances between the old 
    and new levels.

See examples.

## Examples
```julia
using Jchemo

dict = Dict("a" => 1000, "b" => 1, "c" => 2)
x = ["c" ; "c" ; "a" ; "a" ; "a"]
recod_catbydict(x, dict)

x = ["c" ; "c" ; "a" ; "a" ; "a" ; "e"]
recod_catbydict(x, dict)
```
"""
function recod_catbydict(x, dict)
    replace(x, dict...)
end

"""
    recod_catbyind(x, lev)
 Recode a categorical variable to indexes of levels.
* `x` : Categorical variable (n) to replace.
* `lev` : Vector containing categorical levels.

See examples.

*Warning*: The levels in `x` must be contained in `lev`.

## Examples
```julia
using Jchemo

lev = ["EHH" ; "FFS" ; "ANF" ; "CLZ" ; "CNG" ; "FRG" ; "MPW" ; "PEE" ; "SFG" ; "SFG" ; "TTS"]
slev = mlev(lev)
[slev 1:length(slev)] 
x = ["EHH" ; "TTS" ; "FRG" ; "EHH"]
recod_catbyind(x, lev)
```
"""
function recod_catbyind(x, lev)
    lev = mlev(lev)
    xindex = list(Int, length(x))
    @inbounds for i in eachindex(x)
        xindex[i] = findall(lev .== x[i])[1]
    end
    xindex 
end

"""
    recod_catbyint(x; start = 1)
 Recode a categorical variable to integers.
* `x` : Categorical variable (n) to replace.
* `start` : Integer labelling the first categorical level in `x`.

The integers returned by the function correspond to the sorted 
levels of `x`, see examples.

## Examples
```julia
using Jchemo

x = ["b", "a", "b"]
mlev(x)   
[x recod_catbyint(x)]
recod_catbyint(x; start = 0)

recod_catbyint([25, 1, 25])
```
"""
function recod_catbyint(x; start::Int = 1)
    res = dummy(x)
    nlev = length(res.lev)
    u = res.Y .* collect(start:(start + nlev - 1))'
    u = rowsum(u)  
    Int.(u)
end

"""
    recod_catbylev(x, lev)
 Recode a categorical variable to levels.
* `x` : Variable (n) to replace.
* `lev` : Vector containing the categorical levels.

The ith sorted level in `x` is replaced by the ith sorted level in `lev`, 
see examples.

*Warning*: `x` and `lev` must contain the same number of levels.

## Examples
```julia
using Jchemo

x = [10 ; 4 ; 3 ; 3 ; 4 ; 4]
lev = ["B" ; "C" ; "AA" ; "AA"]
mlev(x)
mlev(lev)
[x recod_catbylev(x, lev)]
xstr = string.(x)
[xstr recod_catbylev(xstr, lev)]

lev = [3; 0; 0; -1]
mlev(x)
mlev(lev)
[x recod_catbylev(x, lev)]
```
"""
function recod_catbylev(x, lev)
    n = length(x)
    xlev = mlev(x)
    lev = mlev(lev)
    @assert length(xlev) == length(lev) "x and lev must contain the same number of levels."
    z = similar(lev, n)
    @inbounds for i in eachindex(lev)
        s = findall(x .== xlev[i])
        z[s] .= lev[i] 
    end
    z
end

"""
    recod_indbylev(x::Union{Int, Array{Int}}, lev::Array)
 Recode an index variable to levels.
* `x` : Index variable (n) to replace.
* `lev` : Vector containing the categorical levels.

Assuming slev = 'sort(unique(lev))', each element `x[i]` (i = 1, ..., n) is 
replaced by `slev[x[i]]`, see examples.

*Warning*: Vector `x` must contain integers between 1 and nlev,
where nlev is the number of levels in `lev`. 

## Examples
```julia
using Jchemo

x = [2 ; 1 ; 2 ; 2]
lev = ["B" ; "C" ; "AA" ; "AA"]
mlev(x)
mlev(lev)
[x recod_indbylev(x, lev)]
recod_indbylev([2], lev)
recod_indbylev(2, lev)

x = [2 ; 1 ; 2]
lev = [3 ; 0 ; 0 ; -1]
mlev(x)
mlev(lev)
recod_indbylev(x, lev)
```
"""
function recod_indbylev(x::Union{Int, Array{Int}}, lev::Array)
    n = length(x)
    isa(x, Int) ? x = [x] : x = vec(x)
    lev = mlev(lev)
    v = similar(lev, n)
    @inbounds for i in eachindex(x)
        v[i] = lev[x[i]]
    end
    v
end

"""
    recod_miss(X; miss = nothing)
    recod_miss(df; miss = nothing)
Declare data as missing in a dataset.
* `X` : A dataset (array).
* `miss` : The code used in the dataset to identify the data 
    to be declared as `missing` (of type `Missing`).
Specific for dataframes:
* `df` : A dataset (dataframe).

The case `miss = nothing` has the only action to allow `missing` in `X` or `df`. 

See examples.

## Examples
```julia
using Jchemo, DataFrames

X = hcat(1:5, [0, 0, 7., 10, 1.2])
X_miss = recod_miss(X; miss = 0)

df = DataFrame(i = 1:5, x = [0, 0, 7., 10, 1.2])
df_miss = recod_miss(df; miss = 0)

df = DataFrame(i = 1:5, x = ["0", "0", "c", "d", "e"])
df_miss = recod_miss(df; miss = "0")
```
"""
function recod_miss(X::AbstractArray; miss = nothing)
    X = convert(Matrix{Union{Missing, eltype(X)}}, X)
    if !isnothing(miss)
        replace!(X, miss => missing)
    end
end

function recod_miss(df::DataFrame; miss = nothing)
    df = allowmissing(df)
    if !isnothing(miss)
        for col in eachcol(df)
            replace!(col, miss => missing)
        end
    end
    df
end

"""
    recod_numbyint(x, q)
 Recode a continuous variable to integers.
* `x` : Continuous variable (n) to replace.
* `q` : Numerical values separating classes in `x`.
    The first class is labelled to 1.  

See examples.

## Examples
```julia
using Jchemo, Statistics
x = [collect(1:10); 8.1 ; 3.1] 

q = [3; 8]
zx = recod_numbyint(x, q)  
[x zx]
probs = [.33; .66]
q = quantile(x, probs) 
zx = recod_numbyint(x, q)  
[x zx]
```
"""
function recod_numbyint(x, q)
    zx = similar(x)
    q = sort(q)
    @inbounds for i in eachindex(x)
        k = 1
        @inbounds for j in eachindex(q)
            x[i] > q[j] ? k = k + 1 : nothing
        end
        zx[i] = k
    end
    Int.(zx)
end

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
    finduniq(id)
Select indexes to make unique the IDs in a ID vector.
* `id` : A vector of IDs.

## Examples
```julia
using Jchemo

id = ["a", "d", "c", "b", "a", "d", "a"]

s = finduniq(id)
id[s]  # unique IDs
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
    rmrow(X::Union{AbstractMatrix, DataFrame}, s::Union{Vector, BitVector, UnitRange, Number})
    rmrow(X::Union{Vector, BitVector}, s::Union{Vector, BitVector, UnitRange, Number})
Remove the rows of a matrix or the components of a vector 
having indexes `s`.
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
    tab(X::DataFrame; vargroup = nothing)
Tabulation of categorical variables.
* `x` : Categorical variable or dataset containing categorical variable(s).
Specific for a dataset:
* `vargroup` : Vector of the names of the group variables to consider 
    in `X` (by default: all the columns of `X`).

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
tab(df; vargroup = [:v1, :v2])
tab(df; vargroup = :v2)
```
"""
tab(X::AbstractArray) = sort(StatsBase.countmap(vec(X)))

function tab(X::DataFrame; vargroup = nothing)
    zX = copy(X)
    isa(zX, Vector) ? zX = DataFrame(x1 = zX) : nothing
    isa(zX, DataFrame) ? nothing : zX = DataFrame(zX, :auto)
    isnothing(vargroup) ? vargroup = names(zX) : nothing
    zX.n = ones(nro(zX))
    res = aggstat(zX; vary = :n, vargroup = vargroup, algo = sum)
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
* `x` : Value to transform.
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
* `x` : Value to transform.
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
* `cols` : Determines the columns of the returned dataframe.
    See ?DataFrames.vcat.

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
View of the j-th column(s) of a matrix `X`, or of the j-th element(s) 
    of vector `x`.
""" 
vcol(X, j) = view(X, :, j)
vcol(x::Vector, i) = view(x, i)
vcol(X::DataFrame, j) = view(Matrix(X), :, j)

"""
    vrow(X::AbstractMatrix, i)
    vrow(X::DataFrame, i)
    vrow(x::Vector, i)
View of the i-th row(s) of a matrix `X`, or of the i-th element(s) 
    of vector `x`.
""" 
vrow(X, i) = view(X, i, :) 
vrow(X::DataFrame, i) = view(Matrix(X), i, :)
vrow(x::Vector, i) = view(x, i)

########### Macros 

""" 
    @mod fun
Shortcut for function `parentmodule`.
* `fun` : The name of a function.

## Examples
```julia
@mod rand
```
"""
macro mod(fun)
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
Display the keyword arguments (with their default values) of a function
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

function  ptype(x)
    println(typeof(x))
    println(size(x))
end

