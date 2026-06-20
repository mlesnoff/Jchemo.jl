"""
    aggstat(X::Matrix{Q}, y::Vector{String}; algo::Function = mean) where Q <: AbstractFloat
    aggstat(dat::DataFrame; sel, group, algo::Function = mean)
Compute column-wise statistics by group in a dataset.
* `X` : Data matrix (n, p).
* `dat` : A dataframe (n, p).
* `y` : A categorical variable (class membership) (n). Must be a `Vector{String}`.
* `algo` : Function to compute (default = mean).
Specific for `X::DataFrame`:
* `sel` : Vector of the names (String or Symbol) of the variables to summarize.
* `group` : Vector of the names (String or Symbol) of the categorical variables defining the groups.

Variables defined in `sel` and `group` must be columns of `X`.

## Examples
```julia
using Jchemo, DataFrames, Statistics

n, p = 20, 5
X = rand(n, p)
datf = DataFrame(X, :auto)
y = string.(rand(1:3, n))

res = aggstat(X, y; algo = sum)
@names res
res.lev 
res.X

aggstat(Matrix(datf), y; algo = sum).X

n, p = 20, 5
X = rand(n, p)
datf = DataFrame(X, string.("v", 1:p))
datf.y1 = rand(1:2, n)
datf.y2 = rand(["a", "b", "c"], n)
datf

aggstat(datf; sel = [:v1, :v2] , group = [:y1, :y2], algo = var)  # return a dataframe 
```
""" 
function aggstat(X::Matrix{Q}, y::Vector{String}; algo::Function = mean) where Q <: AbstractFloat
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

function aggstat(dat::DataFrame; sel::Q, group::Q, algo::Function = mean) where Q <: Union{Vector{String}, Vector{Symbol}}
    gdf = groupby(dat, group) 
    res = combine(gdf, sel .=> algo, renamecols = false)
    sort!(res, group)
end

"""
    aggmean(X::Matrix{Q}, y::Vector{String}) where Q <: AbstractFloat
Compute column-wise means by group in a dataset.
* `X` : Data matrix (n, p).
* `y` : A categorical variable (class membership) (n). Must be a `Vector{String}`.

This is a (faster) particular case of `aggstat`: computes means from a single group variable. 

## Examples
```julia
using Jchemo

n, p = 20, 5
X = rand(n, p)
datf = DataFrame(X, :auto) 
y = string.(rand(1:3, n))

res = aggmean(X, y)
res.X
res.lev 

aggmean(Matrix(datf), y).X
```
""" 
function aggmean(X::Matrix{Q}, y::Vector{String}) where Q <: AbstractFloat
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
    aggsumv(x::Vector{Q}, y::Vector{String}) where Q <: Real
Compute the sum by group over a categorical variable.
* `x` : A vector representing the quantitative variable to sum (n) 
* `y` : A categorical variable (class membership) (n). Must be a `Vector{String}`.

## Examples
```julia
using Jchemo

x = ones(1000)
#x = ones(Int, 1000)
y = vcat(rand(["a" ; "c"], 900), fill("b", 100))

aggsumv(x, y)
```
"""
function aggsumv(x::Vector{Q}, y::Vector{String}) where Q <: Real  # 'Real' for 'Int'
    lev = mlev(y)
    v = similar(x, length(lev)) 
    @inbounds for i in eachindex(lev) 
        s = y .== lev[i]
        v[i] = sum(vrow(x, s))
    end
    (val = v, lev)
end

"""
    dupl(X; digits::Int = 3)
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
function dupl(X; digits::Int = 3)
    X = round.(ensure_mat(X))
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
ensure_df(X::Matrix) = DataFrame(X, :auto)
ensure_df(X::AbstractMatrix) = DataFrame(X, :auto)
ensure_df(X::Vector) = DataFrame([X], :auto)

"""
    ensure_mat(X)
Reshape `X` to a matrix if necessary.
"""
ensure_mat(X::Matrix) = X
ensure_mat(X::AbstractMatrix) = Matrix(X)
ensure_mat(X::Vector) = reshape(X, :, 1)
ensure_mat(X::Number) = reshape([X], 1, 1)
ensure_mat(X::DataFrame) = Matrix(X)

"""
    ensure_mat_mb(Xbl)
Reshape a vector of X data to a vector of matrices if necessary.
"""
ensure_mat_mb(Xbl) = [ensure_mat(Xbl[k]) for k in eachindex(Xbl)]

"""
    findmax_cla(x::Vector{String})
    findmax_cla(x::Vector{String}, v::Vector{Q})
Find the most occurent level in `x`.
* `x` : A categorical variable (class membership) (n). Must be a `Vector{String}`.
* `v` : A quantitative variable on which is computed the occurency of the `x` levels.

If ex-aequos, the function returns the first.

## Examples
```julia
using Jchemo

x = string.(rand(1:3, 10))
tab(x)
findmax_cla(x)
```
"""
findmax_cla(x::Vector{String}) = findmax_cla(x, ones(length(x))) 

function findmax_cla(x::Vector{String}, v::Vector{Q}) where Q <: Real
    res = aggsumv(v, x)
    res.lev[argmax(res.val)]   # if equal, argmax takes the first   
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
    finduniq(x)
Find the first indexes of a vector making unique the levels in this vector.
* `x` : A categorical variable (n) (e.g. IDs). Must be a `Vector{String}`.

Can be used to remove duplicated rows (for instance, identified by a single ID variable) in a dataset.

## Examples
```julia
using Jchemo

v = ["a", "d", "c", "b", "a", "d", "a"]  # a vector of IDs

s = finduniq(v)  # first indexes of v making v without duplicates
v[s]  
```
"""
function finduniq(x::Vector{String})
    n = length(x)
    res = tabdupl(x)
    xd = res.keys
    s = list(Int, 0) 
    for i in eachindex(xd)
        zs = findall(x .== xd[i])
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
    list(Q::Union{DataType, UnionAll}, n::Integer)
Create a Vector{Q}(undef, n).

`isassigned(object, i)` can be used to check if cell i is empty.

## Examples
```julia
using Jchemo

list(Float64, 5)
list(Array{Float64}, 5)
list(Matrix{Int}, 5)
```
"""  
list(Q::Union{DataType, UnionAll}, n::Integer) = Vector{Q}(undef, n)

""" 
    mlev(x::Array{String})
Return the sorted levels of an array or dataset.
* `X` : A categorical array (class membership). Must be of type `String`.
* `datf` : A dataframe.

## Examples
```julia
using Jchemo

x = rand(["a";"b";"c"], 20)
lev = mlev(x)
nlev = length(lev)

X = reshape(x, 5, 4)
mlev(X)

datf = DataFrame(g1 = rand(1:2, n), g2 = rand(["a"; "c"], n))
mlev(datf)
```
"""
mlev(x::Array{String}) = sort(unique(x)) 

mlev(datf::DataFrame) = sort(unique(datf)) 

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
    out(x::Vector{Q}, y::Vector{Q}) where Q <: AbstractFloat
Return if elements of a vector are strictly outside of a given range.
* `x` : A quantititative variable whose each element is evaluated to be out of or in the range 
    (min, max) defined from `y`.
* `y` : A quantititative variable on which is computed the range (min, max).

Return a BitVector.

## Examples
```julia
using Jchemo

x = [-200.; -100; -1; 0; 1; 200]
out(x, [-1; .2; 1])
out(x, [-1., 1])
```
"""
out(x::Vector{Q}, y::Vector{Q}) where Q <: AbstractFloat = (x .< minimum(y)) .| (x .> maximum(y))

"""
    pval(d::Distribution, q::Union{Q, Vector{Q}}) where Q <:AbstractFloat
    pval(x::AbstractVector{Q}, q::Union{Q, Vector{Q}}) where Q <:AbstractFloat
    pval(e_cdf::ECDF, q::Union{Q, Vector{Q}}) where Q <:AbstractFloat
Compute p-value(s) from a distribution, an ECDF or a vector.
* `d` : A distribution computed from `Distribution.jl`.
* `x` : A quantitative variable.
* `e_cdf` : An ECDF computed from `StatsBase.jl`.
* `q` : Value(s) (quantile of the considered distribution) for which to compute the p-value(s).

Compute or estimate the p-value of quantile `q`, ie. V(Q > `q`) where Q is the random variable.

## Examples
```julia
using Jchemo, Distributions

d = Distributions.Normal(0, 1)
q = 1.96
#q = [1.64; 1.96]
Distributions.cdf(d, q)    # cumulative density function (CDF)
Distributions.ccdf(d, q)   # complementary CDF (CCDF)
pval(d, q)                 # Distributions.ccdf

x = rand(5)
e_cdf = Jchemo.ecdf(x)
e_cdf(x)                # empirical CDF computed at each point of x (ECDF)
p_val = 1 .- e_cdf(x)   # complementary ECDF at each point of x
q = .3
#q = [.3; .5; 10]
pval(e_cdf, q)          # = 1 .- e_cdf(q)
pval(x, q)
```
"""
pval(d::Distribution, q::Union{Q, Vector{Q}}) where Q <:AbstractFloat = Distributions.ccdf(d, q)

pval(e_cdf::ECDF, q::Union{Q, Vector{Q}}) where Q <:AbstractFloat = 1 .- e_cdf(q)

pval(x::AbstractVector{Q}, q::Union{Q, Vector{Q}}) where Q <: AbstractFloat = pval(StatsBase.ecdf(x), q)

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
    rmcol(X::Union{AbstractMatrix, DataFrame}, s::Union{Int, BitVector, Vector{Int}, UnitRange})
    rmcol(x::Vector, s::Union{Int, BitVector, Vector{Int}, UnitRange})
Remove the columns of a matrix or the components of a vector 
having indexes `s`.
* `X` : A data set (n, p).
* `x` : A variable (n).
* `s` : Vector of the indexes.

## Examples
```julia
using Jchemo

X = rand(5, 3) 
rmcol(X, [1, 3])
rmcol(X, 1:2)

x = rand(5)
rmcol(x, [1, 3])
```
"""
function rmcol(X::Union{AbstractMatrix, DataFrame}, s::Union{Int, BitVector, Vector{Int}, UnitRange})
    if isa(s, BitVector) ; s = findall(s .== 1) ; end
    X[:, setdiff(1:end, s)]
end

function rmcol(x::Vector, s::Union{Int, BitVector, Vector{Int}, UnitRange})
    if isa(s, BitVector) ; s = findall(s .== 1) ; end
    x[setdiff(1:end, s)]
end

"""
    rmrow(X::Union{AbstractMatrix, DataFrame}, s::Union{Int, BitVector, Vector{Int}, UnitRange})
    rmrow(x::Vector, s::Union{Int, BitVector, Vector{Int}, UnitRange})
Remove the rows of a matrix or the components of a vector having indexes `s`.
* `X`, `x` : Matrix and vector, respectively.
* `s` : Vector of the indexes.

## Examples
```julia
using Jchemo

X = rand(5, 3) 
rmrow(X, [1, 3])
rmrow(X, 1:2)

x = rand(5)
rmrow(x, [1, 3])
```
"""
function rmrow(X::Union{AbstractMatrix, DataFrame}, s::Union{Int, BitVector, Vector{Int}, UnitRange})
    if isa(s, BitVector) ; s = findall(s .== 1) ; end
    X[setdiff(1:end, s), :]
end

rmrow(x::Vector, s::Union{Int, BitVector, Vector{Int}, UnitRange}) = rmcol(x, s)

"""
    softmax(x::AbstractVector{Q}) where Q <: AbstractFloat
    softmax(X::AbstractMatrix{Q})  where Q <: AbstractFloat
Softmax transformation.
* `x` : A quantitative variable to transform.
* `X` : A quantitative matrix whose rows are transformed.

Let v be a vector:
* 'softmax'(v) = exp.(v) / sum(exp.(v))

## Examples
```julia
using Jchemo

x = rand(3)
softmax(x)

X = rand(5, 3)
softmax(X)
```
"""
function softmax(x::AbstractVector{Q}) where Q <: AbstractFloat
    expx = exp.(x) 
    expx / sum(expx)
end

function softmax(X::AbstractMatrix{Q})  where Q <: AbstractFloat
    V = similar(X)
    @inbounds for i in axes(X, 1)
        V[i, :] .= softmax(vrow(X, i))
    end
    V
end

"""
    sourcedir(path::String)
Include all the files contained in a directory.
"""
function sourcedir(path::String)
    z = readdir(path)  ## List of files in path
    for i in eachindex(z)
        include(string(path, "/", z[i]))
    end
end

"""
    summ(X; digits::Int = 3)
    summ(X, y::Vector{String}; digits::Int = 3)
Summarize a variable or a dataset.
* `X` : A variable (n) or dataset (n, p).
* `y` : A categorical variable (class membership) (n). Must be a `Vector{String}`.
* `digits` : Nb. digits in the outputs.

## Examples
```julia
using Jchemo

n = 50
x = rand(50)
res = summ(x) ;
@names res
res.ntot
res.res 

X = rand(n, 3) 
summ(X).res

y = string.(rand(1:3, n))
summ(X, y)
```
"""
function summ(X; digits::Int = 3)
    X = ensure_df(X)
    res = StatsBase.describe(X, :mean, :std, :min, :max, :nmissing) 
    insertcols!(res, 6, :n => nro(X) .- res.nmissing)
    for i = 2:4
        z = vcol(res, i)
        s = findall(isa.(z, Real))
        res[s, i] .= round.(res[s, i]; digits)
    end
    (res = res, ntot = nro(X))
end

function summ(X, y::Vector{String}; digits::Int = 3)
    lev = mlev(y)
    for i in eachindex(lev)
        s = y .== lev[i]
        res = summ(X[s, :]; digits).res
        println("Class: ", lev[i])
        println(res)
        println("") ; println("") 
    end
end

"""
    thresh_hard(x::Q, delta::Q) where Q <: AbstractFloat
Hard thresholding function.
* `x` : Value (scalar) to transform.
* `delta` : Limit for the thresholding.

The returned value is:
* abs(`x`) > `delta` ? `x` : 0
where delta >= 0.

## Examples
```julia
using Jchemo, CairoMakie 

delta = .7
thresh_hard(.1, delta)
thresh_hard(3., delta)

x = LinRange(-2, 2, 500)
y = thresh_hard.(x, delta)
lines(x, y; axis = (xlabel = "x", ylabel = "f(x)"))
```
"""
function thresh_hard(x::Q, delta::Q) where Q <: AbstractFloat
    @assert delta >= 0 "delta must be >= 0."
    if abs(x) <= delta
        x = zero(eltype(x)) 
    end
    x
end


"""
    thresh_soft(x::Q, delta::Q) where Q <: AbstractFloat
Soft thresholding function.
* `x` : Value (scalar) to transform.
* `delta` : Limit for the thresholding.

The returned value is:
* sign(`x`) * max(0, abs(`x`) - `delta`)
where delta >= 0.

## Examples
```julia
using Jchemo, CairoMakie 

delta = .7
thresh_soft(.1, delta)
thresh_soft(3., delta)

x = LinRange(-2, 2, 500)
y = thresh_soft.(x, delta)
lines(x, y; axis = (xlabel = "x", ylabel = "f(x)"))
```
"""
function thresh_soft(x::Q, delta::Q) where Q <: AbstractFloat
    @assert delta >= 0 "delta must be >= 0."
    ## same as: abs(x) > delta ? sign(x) * (abs(x) - delta) : zero(eltype(x))
    sign(x) * max(zero(eltype(x)), abs(x) - delta)  # type consistent
end

"""
    vcatdf(dat::Vector{DataFrame}; 
        cols::Union{Q, Vector{String}, Vector{Q}} = :intersect) where Q <: Symbol
Vertical concatenation of a list of dataframes.
* `dat` : List (vector) of dataframes.
* `cols` : Determines the columns of the returned dataframe. See ?DataFrames.vcat.

## Examples
```julia
using Jchemo, DataFrames

dat1 = DataFrame(rand(5, 2), [:v3, :v1]) 
dat2 = DataFrame(100 * rand(2, 2), [:v3, :v1])
dat = [dat1, dat2]
res = vcatdf(dat) ;
@names res
res.X

dat2 = DataFrame(100 * rand(2, 2), [:v1, :v3])
dat = [dat1, dat2]
vcatdf(dat)

dat2 = DataFrame(100 * rand(2, 3), [:v3, :v1, :a])
dat = [dat1, dat2]
vcatdf(dat)
vcatdf(dat; cols = :union)
vcatdf(dat; cols = :intersect)
```
""" 
function vcatdf(dat::Vector{DataFrame}; 
        cols::Union{Q, Vector{String}, Vector{Q}} = :intersect) where Q <: Symbol
    n = length(dat) 
    X = copy(dat[1])
    group = fill(1, nro(X))
    if n > 1
        for i = 2:n
            X = DataFrames.vcat(X, dat[i]; cols = cols)
            group = vcat(group, fill(i, nro(dat[i])))
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
    esc( :( dump(Jchemo.defaults($fun)()) ))
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

