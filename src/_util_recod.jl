"""
    recod_catbydict(x, dict)
 Recode a categorical variable to dictionnary levels.
* `x` : Categorical variable (n) to replace.
* `dict` : Dictionary giving the correpondances between the old and new levels.

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
 Recode a categorical variable to indexes of sorted levels.
* `x` : Categorical variable (n) to replace.
* `lev` : Vector containing categorical levels. 

Internally in the function, the elements of vector `lev` are made unique and are sorted.

See examples.

*Warning*: The levels in `x` must be contained in `lev`.

## Examples
```julia
using Jchemo

lev = ["EHH" ; "FFS" ; "ANF" ; "CLZ" ; "CNG" ; "FRG" ; "MPW" ; "PEE" ; "SFG" ; "SFG" ; "TTS"]
lev_sorted = mlev(lev)
[lev_sorted 1:length(lev_sorted)] 
x = ["EHH" ; "TTS" ; "FRG" ; "EHH"]
recod_catbyind(x, lev)
```
"""
recod_catbyind(x, lev) = Int.(indexin(x, mlev(lev)))

"""
    recod_catbyint(x; start = 1)
 Recode a categorical variable to integers.
* `x` : Categorical variable (n) to replace.
* `start` : Integer labelling the first categorical level in `x`.

The integers returned by the function correspond to the sorted levels of `x`, see examples.

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

The ith sorted level in `x` is replaced by the ith sorted level in `lev`, see examples.

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
    recod_contbyint(x, q)
 Recode a continuous variable to integers.
* `x` : Continuous variable (n) to replace.
* `q` : Numerical values separating classes in `x`. The first class is labelled to 1.  

For a given value x of vector `x`, with `q` being of length K: 
* x <= q[1]             : ==> 1
* q[1] < x <= q[2]      : ==> 2
* q[2] < x <= q[3]      : ==> 3
* etc.
* q[K - 1] < x <= q[K]  : ==> K
* q[K] < x              : ==> K + 1 

See examples.

## Examples
```julia
using Jchemo, Statistics
x = [collect(1:10); 8.1 ; 3.1] 

q = [3; 8]
zx = recod_contbyint(x, q)  
[x zx]
probs = [.33; .66]
q = quantile(x, probs) 
zx = recod_contbyint(x, q)  
[x zx]
```
"""
function recod_contbyint(x, q)
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
    recod_indbylev(x::Union{Int, Array{Int}}, lev::Array)
 Recode an index variable to levels.
* `x` : Index variable (n) to replace.
* `lev` : Vector containing the categorical levels.

Assuming lev_sorted = 'sort(unique(lev))', each element `x[i]` (i = 1, ..., n) is replaced by `lev_sorted[x[i]]`, 
see examples.

*Warning*: Vector `x` must contain integers between 1 and nlev, where nlev is the number of levels in `lev`. 

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
* `miss` : The code used in the dataset to identify the data to be declared as `missing` (of type `Missing`).
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
    expand_tab2d(X; namr = nothing, namc = nothing, namv = nothing)
Expand a 2-D contingency table in a dataframe of two categorical variables.
* `X` : 2-D contincency table (m, p).
Keyword arguments:
* `namr` : Vector (m) of names of the `X`-rows. 
* `namc` : Vector (p) of names of the `X-columns.
* `namv` : Vector (2) of the names of the output categorical variables.

The eventual names in `namr` (`namc`) must have the same length and be in the same order as the rows (columns) of `X`.  

## Examples 
```julia
using Jchemo

X = [5 3 4 ; 1 2 3]

expand_tab2d(X)

res = expand_tab2d(X; namr = [:B1, :AA], namc = collect('a':'c'))
tab(string.(res.v1, "-", res.v2))
```
"""
function expand_tab2d(X; namr = nothing, namc = nothing, namv = nothing)
    X = ensure_mat(X) 
    m, p = size(X)
    isnothing(namr) ? namr = collect(1:m) : nothing
    isnothing(namc) ? namc = collect(1:p) : nothing
    isnothing(namv) ? namv = ["v1"; "v2"] : nothing
    res = list(Matrix, m * p)
    k = 1
    for j = 1:p, i = 1:m 
        n = X[i, j]
        res[k] = hcat(repeat([namr[i]], n), repeat([namc[j]], n))
        k += 1
    end
    DataFrame(string.(reduce(vcat, res)), namv)
end

