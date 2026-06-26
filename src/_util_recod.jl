
#### Make a dummy table from a categorical variable

"""
    dummy([Q::DataType], y::Vector{String})
Compute dummy table from a categorical variable.
* `Q` : Data type for the returned output (dummy table) `Y`.
* `y` : A categorical variable (class membership) (n). Must be a `Vector{String}`.

## Examples
```julia
using Jchemo

y = ["d", "a", "b", "c", "b", "c"]
#y =  string.(rand(1:3, 7))
res = dummy(y)
@names res
res.Y

dummy(Float32, y).Y
```
"""
dummy(y::Vector{String}) = dummy(Float64, y)

function dummy(Q::DataType, y::Vector{String})
    lev = mlev(y)
    ## Thanks to the idea given in the following post of @Mattriks:
    ## https://discourse.julialang.org/t/all-the-ways-to-do-one-hot-encoding/64807/4
    Y = Q.(y .== permutedims(lev))
    (Y = Y, lev)
end

#### Expand a 2D contingency table

"""
    expand_tab2d(X::Matrix{Q}; levr::T = nothing, levc::T = nothing, 
        namv::T = nothing) where {Q <: Signed, T <: Union{Nothing, Vector{String}}}
Expand a 2-D contingency table to a dataframe of two categorical variables.
* `X` : 2-D contincency table (m, p).
Keyword arguments:
* `levr` : Vector (m) of names of the `X`-rows. 
* `levc` : Vector (p) of names of the `X`-columns.
* `namv` : Vector (2) of the names of the output categorical variables.

The eventual names in `levr` (or`levc`) must have the same length and be in the same order as the rows 
(or columns) of `X`.  

The levels are returned in String.

## Examples 
```julia
using Jchemo

X = [5 3 4 ; 1 2 3]

expand_tab2d(X)

res = expand_tab2d(X; levr = ["B1", "AA"], levc = string.(collect('a':'c')), namv = ["fact1", "fact2"])

tab(string.(res.fact1, "-", res.fact2))
```
"""
function expand_tab2d(X::Matrix{Q}; levr::T = nothing, levc::T = nothing, 
        namv::T = nothing) where {Q <: Signed, T <: Union{Nothing, Vector{String}}}
    m, p = size(X)
    if isnothing(levr) ; levr = string.(collect(1:m)) ; end
    if isnothing(levc) ; levc = string.(collect(1:p)) ; end
    if isnothing(namv) ; namv = ["v1"; "v2"] ; end
    res = list(Matrix{String}, m * p)
    k = 1
    for j in axes(X, 2), i in axes(X, 1) 
        n = X[i, j]
        res[k] = hcat(fill(levr[i], n), fill(levc[j], n))
        k += 1
    end
    DataFrame(reduce(vcat, res), namv)
    #DataFrame(string.(reduce(vcat, res)), namv)
end

"""
    expand_grid(; kwargs...)
Build a dataframe with all the combinations of the entered parameter values.
Keyword arguments:
* `kwargs` : Named vector(s) of the parameter(s) values.

## Examples 
```julia
using Jchemo

expand_grid(y1 = [1, 2], y2 = ["A"; "B"; "C"], y3 = 15.5)

expand_grid(y1 = (1, 2), y2 = ["A", "B", "C"], y3 = (:ab,))
```
"""
function expand_grid(; kwargs...)
    pars = mpar(; kwargs...) 
    typ = [typeof(pars[i][1]) for i in eachindex(pars)]
    res = DataFrame(reduce(hcat, pars), collect(@names pars))
    convertdf(res, typ)
end

## Not exported

## v = (y1 = [1, 2], y2 = string.(collect('a':'c')), y3 = [:a1])  # the function does not accept thype 'Char'
## Jchemo.mpar_tupl(v) 
## Jchemo.expand_grid_tupl(v)
function expand_grid_tupl(tupl::NamedTuple)
    pars = Jchemo.mpar_tupl(tupl) 
    typ = [typeof(pars[i][1]) for i in eachindex(pars)]
    println(typ)
    println(collect(@names pars))
    res = DataFrame(reduce(hcat, pars), collect(@names pars))
    convertdf(res, typ)
end

## ind = [[[1, 2]]; [[1]]; [[1, 2, 3]]]
## Jchemo.indcumul(ind)
function indcumul(v::Vector{Vector{Int}})
    ind = 1
    res = Vector{Vector{Int}}()
    for u in v
        push!(res, collect(ind:ind+length(u) - 1))
        ind += length(u)
    end
    res
end

#### Recode a single variable

"""
    recod_catbydict(x::Vector{Q}, dict::Dict{Q, Q}) where Q <: String
Recode a categorical variable by levels defined in a dictionnary.
* `x` : A categorical variable (class membership) (n). Must be a `Vector{String}`.
* `dict` : Dictionary giving the correpondances between the old and new (String) levels.

See examples.

## Examples
```julia
using Jchemo

dict = Dict("a" => "1000", "b" => "1", "c" => "2")
x = ["c" ; "c" ; "a" ; "a" ; "a"]
recod_catbydict(x, dict)

x = ["c" ; "c" ; "a" ; "a" ; "a" ; "e"]
recod_catbydict(x, dict)
```
"""
function recod_catbydict(x::Vector{Q}, dict::Dict{Q, Q}) where Q <: String
    Q.(replace(x, dict...))
end

"""
    recod_catbyind(x::Vector{Q}, lev::Vector{Q}) where Q <: String = Int.(indexin(x, mlev(lev)))
Recode a categorical variable by indexes of sorted levels.
* `x` : A categorical variable (class membership) (n). Must be a `Vector{String}`.
* `lev` : Vector containing the new categorical levels. Must be a `Vector{String}`. 

Replace element `x[i]` (i = 1, ..., n) by the index of the level in `lev` corresponding to `x[i]`, 
see examples. Internally, the function makes unique and sorts the elements of vector `lev`.

*Warning*: All levels in `x` must be contained in `lev`.

## Examples
```julia
using Jchemo

lev = ["EHH" ; "FFS" ; "ANF" ; "CLZ" ; "CNG" ; "FRG" ; "MPW" ; "PEE" ; "SFG" ; "SFG" ; "TTS"]
levsorted = mlev(lev)
[levsorted 1:length(levsorted)] 
x = ["EHH" ; "TTS" ; "FRG" ; "EHH"]
recod_catbyind(x, lev)
```
"""
recod_catbyind(x::Vector{Q}, lev::Vector{Q}) where Q <: String = Int.(indexin(x, mlev(lev)))

"""
    recod_catbyind2(x::Vector{Q}; start::Int = 1) where Q <: String
Recode a categorical variable by successive integer indexes.
* `x` : A categorical variable (class membership) (n). Must be a `Vector{String}`.
* `start` : Integer labelling the first categorical level in `x`.

The levels contained in `x` are made unique and sorted, and stored in a vector v. If length(v) = k,
an index vector, vind, is built such as:
* vind = [`start`; `start` + 1; ...; `start` + k - 1] 
The function replaces element `x[i]` by the index in vind corresponding to `x[i]`, see examples.

## Examples
```julia
using Jchemo

x = ["b", "a", "b"]
mlev(x)   
[x recod_catbyind2(x)]
recod_catbyind2(x; start = 0)

recod_catbyind2(string.([25, 1, 25]))
```
"""
function recod_catbyind2(x::Vector{Q}; start::Int = 1) where Q <: String
    res = dummy(x)
    nlev = length(res.lev)
    u = res.Y .* collect(start:(start + nlev - 1))'
    u = rowsum(u)  
    Int.(u)
end

"""
    recod_catbylev(x::Vector{Q}, lev::Vector{Q}) where Q <: String
Recode a categorical variable by levels.
* `x` : A categorical variable (class membership) (n). Must be a `Vector{String}`.
* `lev` : Vector containing the categorical levels. Must be a `Vector{String}`.

The ith sorted level in `x` (i = 1, ..., n) is replaced by the ith sorted level in `lev`, see examples.

*Warning*: `x` and `lev` must contain the same number of levels.

## Examples
```julia
using Jchemo

x = string.([10; 4; 3; 3; 4; 4])
lev = ["B"; "C"; "AA"; "AA"]
mlev(x)
mlev(lev)
[x recod_catbylev(x, lev)]

lev = string.([3; 0; 0; -1])
mlev(x)
mlev(lev)
[x recod_catbylev(x, lev)]
```
"""
function recod_catbylev(x::Vector{Q}, lev::Vector{Q}) where Q <: String
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
    recod_indbylev(x::Vector{Q}, lev::Vector{String}) where Q <: Signed
Recode an index variable by levels.
* `x` : Index variable (n) to recode. Must be a `Vector{Signed}`.
* `lev` : Vector containing categorical levels. Must be a `Vector{String}`.

Assuming levsorted = 'sort(unique(lev))', each element `x[i]` (i = 1, ..., n) is replaced by `levsorted[x[i]]`, 
see examples.

*Warning*: Vector `x` must contain integers withinh [1, nlev], where nlev is the number of levels in `lev`. 

## Examples
```julia
using Jchemo

x = [2; 1; 2; 2]
lev = ["B"; "C"; "AA"; "AA"]
mlev(lev)
[x recod_indbylev(x, lev)]
recod_indbylev([2], lev)

x = [2 ; 1 ; 2]
lev = ["d" ; "a" ; "a" ; "b"]
mlev(lev)
recod_indbylev(x, lev)
```
"""
function recod_indbylev(x::Vector{Q}, lev::Vector{String}) where Q <: Signed
    n = length(x)
    lev = mlev(lev)
    v = similar(lev, n)
    @inbounds for i in eachindex(x)
        v[i] = lev[x[i]]
    end
    v
end

"""
    recod_contbylev(x::Vector{Q}, q::Vector{Q}) where Q <: Float
Recode a quantitative variable by successive levels.
* `x` : A quantitative variable (n) to recode.
* `q` : A vector (K) of the values separating the class levels from `x`.  

The function potentially returns K + 1 levels. For a given value x of vector `x`, and for `q` a vector 
of length K: 
* x <= q[1]             : ==> "1"
* q[1] < x <= q[2]      : ==> "2"
* q[2] < x <= q[3]      : ==> "3"
* etc.
* q[K - 1] < x <= q[K]  : ==> "K"
* q[K] < x              : ==> "K + 1" 

## Examples
```julia
using Jchemo, Statistics
x = [collect(1:10); 8.1 ; 3.1] 

q = [3.; 8]
zx = recod_contbylev(x, q)  
[x zx]
probs = [.33; .66]
q = quantile(x, probs) 
zx = recod_contbylev(x, q)  
[x zx]
tab(zx)
```
"""
function recod_contbylev(x::Vector{Q}, q::Vector{Q}) where Q <: Float
    v = zeros(Int, length(x))
    q = sort(q)
    @inbounds for i in eachindex(x)
        k = 1
        @inbounds for j in eachindex(q)
            if x[i] > q[j]  
                k += 1
            end
        end
        v[i] = k
    end
    string.(v)
end

############ Other recoding

### Missing data  

""" 
    parsemiss(Q, x::Vector{Union{String, Missing}})
Parsing a string vector containing missing data.
* `Q` : Type that results from the parsing of type `String'. 
* `x` : A string vector containing observations `missing` (of type `Missing`).

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
        if !ismissing(x[i])
            v[i] = parse(Q, x[i])
        end
    end
    v
end

"""
    recod_miss(X; miss = nothing)
    recod_miss(datf; miss = nothing)
Declare data as missing in a dataset.
* `X` : An array-dataset.
* `datf` : A dataframe.
* `miss` : The value used in the dataset to identify the missing data. 

Each cell of `X` or `datf` having the value `miss` is replaced by value `missing` of type `Missing`.

The case `miss = nothing` has the only action to allow `missing` in `X` or `datf`. 

See examples.

## Examples
```julia
using Jchemo, DataFrames

X = hcat(1:5, [0, 0, 7., 10, 1.2])
X_miss = recod_miss(X; miss = 0)

datf = DataFrame(i = 1:5, x = [0, 0, 7., 10, 1.2])
datf_miss = recod_miss(datf; miss = 0)

datf = DataFrame(i = 1:5, x = ["0", "0", "c", "d", "e"])
datf_miss = recod_miss(datf; miss = "0")
```
"""
function recod_miss(X::AbstractArray; miss = nothing)
    X = convert(Matrix{Union{Missing, eltype(X)}}, X)
    if !isnothing(miss)
        replace!(X, miss => missing)
    end
    X
end

function recod_miss(datf::DataFrame; miss = nothing)
    datf = allowmissing(datf)
    if !isnothing(miss)
        for col in eachcol(datf)
            replace!(col, miss => missing)
        end
    end
    datf
end


### Convertdf

""" 
    convertdf(datf::DataFrame, typ; miss = nothing)
Convert the columns of a dataframe to given types.
* `datf` : A dataframe.
* `typ` : A vector of the targeted types for the columns of the new dataframe.
Keyword arguments:
* `miss` : The code used in `datf` to identify the data to be declared as `missing` (of type `Missing`).
    See function `recod_miss`.

## Examples
```julia
using Jchemo, DataFrames

datf = DataFrame(y1 = ["1", "2", "3"], y2 = Any["A"; "B"; "C"], y3 = [15.5, 0.2, 1.3])
typ = [Int, String, Float32]
convertdf(datf, typ)

datf = DataFrame(y1 = ["1", "2", "00"], y2 = Any["00"; "00"; "C"], y3 = [15.5, 0.2, 1.3])
typ = [Int, String, Float32]
convertdf(datf, typ; miss = "00")
```
"""
function convertdf(datf::DataFrame, typ::Vector{DataType}; miss = nothing)
    dat = string.(datf)
    dat = recod_miss(dat; miss = string(miss)) 
    res = DataFrame()
    for i in eachindex(typ)
        z = dat[:, i]
        if typ[i] == String
            if sum(ismissing.(z)) == 0 ; z = string.(z) ; end
        elseif typ[i] == Symbol
            if sum(ismissing.(z)) == 0 ; z = Symbol.(z) ; end
        elseif in(typ[i], [Integer, Int32, Int64]) 
            if sum(ismissing.(z)) == 0
                z = convert.(typ[i], parse.(Float64, z))
            else
                ## this case does not work to change floats "1." to int (to do) 
                z = parsemiss(typ[i], z)
            end
        else
            if sum(ismissing.(z)) == 0
                z = parse.(typ[i], z)
            else
                z = parsemiss(typ[i], z)
            end
        end
        res = hcat(res, z; makeunique = true)
    end
    rename!(res, names(dat))
    res
end

