"""
    aggstat(X, y; algo = mean)
    aggstat(X::DataFrame; vars, groups, algo = mean)
Compute column-wise statistics by class in a dataset.
* `X` : Data (n, p).
* `y` : A categorical variable (n) (class membership).
* `algo` : Function to compute (default = mean).
Specific for dataframes:
* `vars` : Vector of the ames of the variables to summarize.
* `groups` : Vector of the names of the categorical variables to consider
    for computations by class.

Variables defined in `vars` and `groups` must be columns of `X`.

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
aggstat(df; vars = [:v1, :v2], groups = [:gr1, :gr2], algo = var)
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
        zX[i, j] = algo(X[s, j])
    end
    (X = zX, lev)
end

function aggstat(X::DataFrame; vars, groups, algo = mean)
    gdf = groupby(X, groups) 
    res = combine(gdf, vars .=> algo, renamecols = false)
    sort!(res, groups)
end

""" 
    aggsum(x::Vector, y::Vector)
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
function aggsum(x::Vector, y::Vector)
    lev = mlev(y)
    v = similar(x, length(lev)) 
    @inbounds for i in eachindex(lev) 
        s = y .== lev[i]
        v[i] = sum(vrow(x, s))
    end
    v
end

"""
    corm(X, weights::Weight)
    corm(X, Y, weights::Weight)
Compute a weighted correlation matrix.
* `X` : Data (n, p).
* `Y` : Data (n, q).
* `weights` : Weights (n) of the observations.
    Object of type `Weight` (e.g. generated by 
    function `mweight`).

Uncorrected correlation matrix 
* of `X`-columns :  ==> (p, p) matrix 
* or between `X`-columns and `Y`-columns :  ==> (p, q) matrix.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
Y = rand(n, 3)
w = mweight(rand(n))

corm(X, w)
corm(X, Y, w)
```
"""
function corm(X, weights::Weight)
    zX = copy(ensure_mat(X))
    xmeans = colmean(zX, weights)
    xstds = colstd(zX, weights)
    fcenter!(zX, xmeans)
    fscale!(zX, xstds)
    z = Diagonal(sqrt.(weights.w)) * zX
    z' * z
end

function corm(X, Y, weights::Weight)
    zX = copy(ensure_mat(X))
    zY = copy(ensure_mat(Y))
    xmeans = colmean(zX, weights)
    ymeans = colmean(zY, weights)
    xstds = colstd(zX, weights)
    ystds = colstd(zY, weights)
    fcenter!(zX, xmeans)
    fcenter!(zY, ymeans)
    fscale!(zX, xstds)
    fscale!(zY, ystds)
    zX' * Diagonal(weights.w) * zY
end

"""
    cosm(X)
    cosm(X, Y)
Compute a cosinus matrix.
* `X` : Data (n, p).
* `Y` : Data (n, q).

The function computes the cosinus matrix: 
* of the columns of `X`:  ==> (p, p) matrix 
* or between columns of `X` and `Y` :  ==> (p, q) matrix.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
Y = rand(n, 3)

cosm(X)
cosm(X, Y)
```
"""
function cosm(X)
    X = ensure_mat(X)
    xnorms = colnorm(X)
    zX = fscale(X, xnorms)
    zX' * zX 
end

function cosm(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    xnorms = colnorm(X)
    ynorms = colnorm(Y)
    zX = fscale(X, xnorms)
    zY = fscale(Y, ynorms)
    zX' * zY 
end

"""
    cosv(x, y)
Compute cosinus between two vectors.
* `x` : vector (n).
* `y` : vector (n).

## Examples
```julia
using Jchemo

n = 5
x = rand(n)
y = rand(n)

cosv(x, y)
```
"""
cosv(x, y) = dot(x, y) / (norm(x) * norm(y))


"""
    covm(X, weights::Weight)
    covm(X, Y, weights::Weight)
Compute a weighted covariance matrix.
* `X` : Data (n, p).
* `Y` : Data (n, q).
* `weights` : Weights (n) of the observations.
    Object of type `Weight` (e.g. generated by 
    function `mweight`).

The function computes the uncorrected weighted covariance 
matrix: 
* of the columns of `X`:  ==> (p, p) matrix 
* or between columns of `X` and `Y` :  ==> (p, q) matrix.

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
Y = rand(n, 3)
w = mweight(rand(n))

covm(X, w)
covm(X, Y, w)
```
"""
function covm(X, weights::Weight)
    zX = copy(ensure_mat(X))
    fcenter!(zX, colmean(zX, weights))
    zX = Diagonal(sqrt.(weights.w)) * zX
    zX' * zX
end

function covm(X, Y, weights::Weight)
    zX = copy(ensure_mat(X))
    zY = copy(ensure_mat(Y))
    fcenter!(zX, colmean(zX, weights))
    fcenter!(zY, colmean(zY, weights))
    zX' * Diagonal(weights.w) * zY
end

"""
    dummy(y, T = Float64)
Compute dummy table from a categorical variable.
* `y` : A categorical variable.
* `T` : Type of the output dummy table `Y`.

## Examples
```julia
using Jchemo

y = ["d", "a", "b", "c", "b", "c"]
#y =  rand(1:3, 7)
res = dummy(y)
pnames(res)
res.Y
```
"""
function dummy(y, T = Float64)
    n = length(y)
    lev = mlev(y)
    nlev = length(lev)
    Y = BitArray(undef, n, nlev)  # Type = BitMatrix
    @inbounds for i in eachindex(lev)
        Y[:, i] = y .== lev[i]
    end
    Y = convert.(T, Y)
    (Y = Y, lev)
end

## Not exported (slower)
function dummy2(y)
    lev = mlev(y)
    nlev = length(lev)
    res = list(BitVector, nlev)
    @inbounds for i in eachindex(lev)
        res[i] = y .== lev[i]
    end
    Y = reduce(hcat, res)
    (Y = Y, lev)
end

"""
    dupl(X; digits = 3)
Find duplicated rows in a dataset.
* `X` : A dataset.
* `digits` : Nb. digits used to round `X`
    before checking.

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
    # round, etc. does not
    # accept missing values
    X[ismissing.(X)] .= -1e5
    # End
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
    res = DataFrame((rownum1 = rownum1[u], 
        rownum2 = rownum2[u]))
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
## Tentative to work with CUDA
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
* `weights` : Weights (n) of the observations.
    Object of type `Weight` (e.g. generated by 
    function `mweight`).

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
    frob(X)
    frob(X, weights::Weight)
Frobenius norm of a matrix.
* `X` : A matrix (n, p).
* `weights` : Weights (n) of the observations.
    Object of type `Weight` (e.g. generated by 
    function `mweight`).

The Frobenius norm of `X` is:
* sqrt(tr(X' * X)).

The Frobenius weighted norm is:
* sqrt(tr(X' * D * X)), where D is the diagonal matrix of vector `w`.
"""
frob(X) = LinearAlgebra.norm(X)

frob(X, weights::Weight) = sqrt(sum(weights.w' * (X.^2))) 

"""
    head(X)
Display the first rows of a dataset.

## Examples
```julia
using Jchemo

X = rand(100, 5)
head(X)
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
    esc( :( head($X) ))
end

"""
    iqr(x)
Compute the interquartile interval (IQR).

## Examples
```julia
x = rand(100)
iqr(x)
```
"""
## Not exported
iqr(x) = quantile(x, .75) - quantile(x, .25)

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
list(Q, n::Integer) = Vector{Q}(undef, n)

""" 
    mad(x)
Compute the median absolute deviation (MAD),
adjusted by a factor (1.4826) for asymptotically normal consistency. 

## Examples
```julia
using Jchemo

x = rand(100)
mad(x)
```
"""
## Not exported
mad(x) = 1.4826 * median(abs.(x .- median(x)))

"""
    miss(X)
Find rows with missing data in a dataset.
* `X` : A dataset.

## Examples
```julia
using Jchemo

X = rand(5, 4)
zX = hcat(rand(2, 3), fill(missing, 2))
Z = vcat(X, zX)
miss(X)
miss(Z)
```
"""
function miss(X)
    X = ensure_mat(X)
    z = vec(sum(ismissing.(X); dims = 2))
    u = findall(z .> 0) 
    DataFrame((rownum = u, nmissing = z[u]))
end

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

df = DataFrame(g1 = rand(1:2, n), 
    g2 = rand(["a"; "c"], n))
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
    mweightcla(x::Vector; prior::Union{Symbol, Vector} = :unif)
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
function mweightcla(x::Vector; prior::Union{Symbol, Vector} = :unif)
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

function mweightcla(Q::DataType, x::Vector; prior::Union{Symbol, Vector} = :unif)
    mweight(convert.(Q, mweightcla(x; prior).w))
end

""" 
    nco(X)
Return the nb. columns of `X`.
"""
nco(X) = size(X, 2)

""" 
    normw(x, weights::Weight)
Compute the weighted norm of a vector.
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).

The weighted norm of vector `x` is computed by:
* sqrt(x' * D * x), where D is the diagonal matrix of vector `weights.w`.
"""
normw(x, weights::Weight) = sqrt(sum(x .* weights.w .* x))

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
    plist(x)
Print each element of a list.
"""
function plist(x)
    nam = pnames(x)
    for i in eachindex(nam)
        println("--- ", nam[i])        
        println("")
        println(x[i])
        println("")
    end
end

""" 
    pmod(foo)
Shortcut for function `parentmodule`.
"""
pmod(foo) = parentmodule(foo)

""" 
    pnames(x)
Return the names of the elements of `x`.
"""
pnames(x) = propertynames(x)

""" 
    psize(x)
Print the type and size of `x`.
"""
function  psize(x)
    println(typeof(x))
    println(size(x))
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
ie. P(Q > `q`) where Q is the random variable.

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
Replace a categorical variable by dictionnary levels.
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
Replace a categorical variable by indexes of levels.
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
Replace a categorical variable by integers.
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
    z = dummy(x).Y
    nlev = nco(z)
    u = z .* collect(start:(start + nlev - 1))'
    u = rowsum(u)  
    Int.(u)
end

"""
    recod_catbylev(x, lev)
Replace a categorical variable by levels.
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
Replace an index variable by levels.
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
    recod_numbyint(x, q)
Replace a continuous variable by integers.
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
    rmcol(X, s)
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
function rmcol(X::Union{AbstractMatrix, DataFrame}, 
        s::Union{Vector, BitVector, UnitRange, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[:, setdiff(1:end, Int.(s))]
end

function rmcol(X::Vector, 
        s::Union{Vector, BitVector, UnitRange, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[setdiff(1:end, Int.(s))]
end

"""
    rmrow(X, s)
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
function rmrow(X::Union{AbstractMatrix, DataFrame}, 
        s::Union{Vector, BitVector, UnitRange, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[setdiff(1:end, Int.(s)), :]
end

function rmrow(X::Vector, 
        s::Union{Vector, BitVector, UnitRange, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[setdiff(1:end, Int.(s))]
end

"""
    soft(x::Real, delta)
Soft thresholding function.
* `x` : Value to transform.
* `delta` : Range for the thresholding.

The returned value is:
* sign(x) * max(0, abs(x) - delta)
where delta >= 0.

## Examples
```julia
using Jchemo, CairoMakie 

delta = .2
soft(3, delta)

x = LinRange(-2, 2, 100)
y = soft.(x, delta)
lines(x, y)
```
"""
function soft(x, delta)
    @assert delta >= 0 "delta must be >= 0."
    sign(x) * max(0, abs(x) - delta)
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
    P = similar(X)
    n = nro(P)
    @inbounds for i = 1:n
        P[i, :] .= softmax(vrow(X, i))
    end
    P
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
    ssq(X)
Compute the total inertia of a matrix.
* `X` : Matrix.

Sum of all the squared components of `X` (= `norm(X)^2`; Squared Frobenius norm). 

## Examples
```julia
using Jchemo

X = rand(5, 2) 
ssq(X)
```
""" 
function ssq(X)
    v = vec(ensure_mat(X))
    dot(v, v)
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
pnames(res)
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
        s = findall(isa.(z, Float64))
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
    tab(x)
Univariate tabulation.
* `x` : Categorical variable.

The output cointains sorted levels.

## Examples
```julia
using Jchemo

x = rand(["a";"b";"c"], 20)
res = tab(x)
res.keys
res.vals
```
"""
tab(x) = sort(StatsBase.countmap(vec(x)))

"""
    tabdf(X; groups = nothing)
Compute the nb. occurences in categorical variables of a dataset.
* `X` : Data.
* `groups` : Vector of the names of the group variables to consider 
    in `X` (by default: all the columns of `X`).

The output (dataframe) contains sorted levels.

## Examples
```julia
using Jchemo

n = 20
X =  hcat(rand(1:2, n), rand(["a", "b", "c"], n))
tabdf(X)
tabdf(X[:, 2])

df = DataFrame(X, [:v1, :v2])
tabdf(df)
tabdf(df; groups = [:v1, :v2])
tabdf(df; groups = :v2)
```
""" 
function tabdf(X; groups = nothing)
    zX = copy(X)
    isa(zX, Vector) ? zX = DataFrame(x1 = zX) : nothing
    isa(zX, DataFrame) ? nothing : zX = DataFrame(zX, :auto)
    isnothing(groups) ? groups = names(zX) : nothing
    zX.n = ones(nro(zX))
    res = aggstat(zX; vars = :n, groups = groups, 
        algo = sum)
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
    vcatdf(dat; cols = :intersect) 
Vertical concatenation of a list of dataframes.
* `dat` : List (vector) of dataframes.
* `cols` : Determines the columns of the returned data frame.
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
View of the j-th column(s) of a matrix `X`,
or of the j-th element(s) of vector `x`.
""" 
vcol(X, j) = view(X, :, j)
vcol(x::Vector, i) = view(x, i)
vcol(X::DataFrame, j) = view(Matrix(X), :, j)

"""
    vrow(X::AbstractMatrix, i)
    vrow(X::DataFrame, i)
    vrow(x::Vector, i)
View of the i-th row(s) of a matrix `X`,
or of the i-th element(s) of vector `x`.
""" 
vrow(X, i) = view(X, i, :) 
vrow(X::DataFrame, i) = view(Matrix(X), i, :)
vrow(x::Vector, i) = view(x, i)

##################### MACROS 

"""
    @namvar(x)
Return the name of a variable.
* `x` : A variable or function.

Thanks to: 
https://stackoverflow.com/questions/38986764/save-variable-name-as-string-in-julia

## Examples
```julia
using Jchemo

z = 1:5
Jchemo.@namvar(z)
```
"""
macro namvar(arg)
    x = string(arg)
    quote
        $x
    end
end


