"""
    aggstat(X::Union{AbstractMatrix, AbstractVector}; group, fun = mean)
    aggstat(X::DataFrame; group_nam, var_nam, fun = mean)
Compute the mean (or other statistic) of each column of `X`, by group.
* `X` : A matrix or dataframe (n, p) or vector (n).
* `group` : A vector (n) defining the groups.
* `group_nam` : Names (vector) of the group variables to consider.
* `var_nam` : Names (vector) of the variables to summarize.
* `fun` : Function to compute.

Variables defined in `var_nam` and `group_nam` must be columns of `X`.

## Examples
```julia
n, p = 20, 5
X = rand(n, p)
group = rand(1:3, n)
res = aggstat(X; group = group, fun = sum)
pnames(res)
res.X

n, p = 20, 6
X = DataFrame(rand(n, p), :auto)
X.group1 = rand(1:2, n)
X.group2 = rand(1:3, n)
aggstat(X; var_nam = [:x1, :x2], group_nam = [:group1, :group2], fun = mean)
```
""" 
function aggstat(X::Union{AbstractMatrix, AbstractVector}; group, 
        fun = mean)
    X = ensure_mat(X)
    group = vec(group)
    q = nco(X)
    ztab = tab(group)
    lev = ztab.keys
    nlev = length(lev)
    ni = collect(values(ztab))
    zX = similar(X, nlev, q)
    for i in 1:nlev
        s = group .== lev[i]
        zX[i, :] .= vec(fun(vrow(X, s), dims = 1)) 
    end
    (X = zX, lev = lev, ni = ni)
end

function aggstat(X::DataFrame; group_nam, var_nam, fun = mean)
    gdf = groupby(X, group_nam) 
    res = combine(gdf, var_nam .=> fun, renamecols = false)
    sort!(res, group_nam)
end

"""
    center(X, v)
    center!(X, v)
Center each column of `X`.
* `X` : Data.
* `v` : Centering factors.

## examples
```julia
n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
center(X, xmeans)
```
""" 
function center(X, v)
    zX = copy(ensure_mat(X))
    center!(zX, v)
    zX
end

function center!(X::Matrix, v)
    p = nco(X)
    @inbounds for j = 1:p
        X[:, j] .= vcol(X, j) .- v[j]
    end
end

"""
    cscale(X, u, v)
    cscale!(X, u, v)
Center and scale each column of `X`.
* `X` : Data.
* `u` : Centering factors.
* `v` : Scaling factors.

## examples
```julia
n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
xstds = colstd(X)
cscale(X, xmeans, xstds)
```
""" 
function cscale(X, u, v)
    zX = copy(ensure_mat(X))
    cscale!(zX, u, v)
    zX
end

function cscale!(X::Matrix, u, v)
    p = nco(X)
    @inbounds for j = 1:p
        X[:, j] .= (vcol(X, j) .- u[j]) ./ v[j]
    end
end

# Slower:
function cscale2!(X::Matrix, u, v)
    center!(X, u)
    scale!(X, v)
end

"""
    checkdupl(X; digits = 3)
Find replicated rows in a dataset.
* `X` : A dataset.
* `digits` : Nb. digits used to round `X` before checking.

## Examples
```julia
X = rand(5, 3)
Z = vcat(X, X[1:3, :], X[1:1, :])
checkdupl(X)
checkdupl(Z)

M = hcat(X, fill(missing, 5))
Z = vcat(M, M[1:3, :])
checkdupl(M)
checkdupl(Z)
```
"""
function checkdupl(X; digits = 3)
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
    res = DataFrame((rownum1 = rownum1[u], rownum2 = rownum2[u]))
    res
end

"""
    checkmiss(X)
Find rows with missing data in a dataset.
* `X` : A dataset.

## Examples
```julia
X = rand(5, 4)
zX = hcat(rand(2, 3), fill(missing, 2))
Z = vcat(X, zX)
checkmiss(X)
checkmiss(Z)
```
"""
function checkmiss(X)
    X = ensure_mat(X)
    z = vec(sum(ismissing.(X); dims = 2))
    u = findall(z .> 0) 
    DataFrame((rownum = u, nmissing = z[u]))
end

"""
    colmad(X)
Compute the median absolute deviation (MAD) of each column of `X`.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)

colmad(X)
```
""" 
function colmad(X)
    X = ensure_mat(X)
    p = nco(X)
    z = zeros(p)
    @inbounds for j = 1:p
        z[j] = mad(vcol(X, j))        
    end
    z 
end

"""
    colmean(X)
    colmean(X, w)
Compute the mean of each column of `X`.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.

`w` is internally normalized to sum to 1.

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = collect(1:n)

colmean(X)
colmean(X, w)
```
""" 
colmean(X) = vec(Statistics.mean(ensure_mat(X); dims = 1))

colmean(X, w) = vec(mweight(w)' * ensure_mat(X))

"""
    colnorm(X)
    colnorm(X, w)
Compute the norm of each column of a dataset X.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.

`w` is internally normalized to sum to 1.

The norm of a column x of `X` is:
* sqrt(x' * x), where D is the diagonal matrix of `w`.

The weighted norm is:
* sqrt(x' * D * x), where D is the diagonal matrix of `w`.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = collect(1:n)

colnorm(X)
colnorm(X, w)
```
""" 
function colnorm(X)
    X = ensure_mat(X)
    map(norm, eachcol(X))
end

function colnorm(X, w)
    X = ensure_mat(X)
    #p = nco(X)
    #w = mweight(w)
    #z = similar(X, p)
    #@inbounds for j = 1:p
    #    x = vcol(X, j)
    #    z[j] = sqrt(dot(x, w .* x))
    #end
    #z 
    # Faster:
    sqrt.(mweight(w)' * X.^2)
end

"""
    colstd(X)
    colstd(X, w)
Compute the (uncorrected) standard deviation of each column of `X`.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.

`w` is internally normalized to sum to 1.

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = collect(1:n)

colstd(X)
colstd(X, w)
```
""" 
colstd(X) = sqrt.(colvar(X))

colstd(X, w) = sqrt.(colvar(X, mweight(w)))

"""
    colsum(X)
    colsum(X, w)
Compute the sum of each column of `X`.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.

`w` is internally normalized to sum to 1.

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = collect(1:n)

colsum(X)
colsum(X, w)
```
""" 
colsum(X) = vec(sum(X; dims = 1))

colsum(X, w) = vec(mweight(w)' * ensure_mat(X))

"""
    colvar(X)
    colvar(X, w)
Compute the (uncorrected) variance of each column of `X`.
* `X` : Data (n, p).
* `w` : Weights (n) of the observations.

`w` is internally normalized to sum to 1.

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
w = collect(1:n)

colvar(X)
colvar(X, w)
```
""" 
colvar(X) = vec(Statistics.var(ensure_mat(X); corrected = false, dims = 1))

function colvar(X, w)
    X = ensure_mat(X)
    p = nco(X)
    w = mweight(w)
    z = colmean(X, w)
    @inbounds for j = 1:p
        z[j] = dot(w, (vcol(X, j) .- z[j]).^2)        
    end
    z 
end

"""
    corm(X, w)
    corm(X, Y, w)
Compute correlation matrices.
* `X` : Data (n, p).
* `Y` : Data (n, q).
* `w` : Weights (n) of the observations.

`w` is internally normalized to sum to 1.

Uncorrected correlation matrix 
* of the columns of `X`: ==> (p, p) matrix 
* or between columns of `X` and `Y` : ==> (p, q) matrix.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
Y = rand(n, 3)
w = collect(1:n)

corm(X, w)
corm(X, Y, w)
```
"""
function corm(X, w)
    zX = copy(ensure_mat(X))
    w = mweight(w)
    xmeans = colmean(zX, w)
    xstds = colstd(zX, w)
    center!(zX, xmeans)
    scale!(zX, xstds)
    z = Diagonal(sqrt.(w)) * zX
    z' * z
end

function corm(X, Y, w)
    zX = copy(ensure_mat(X))
    zY = copy(ensure_mat(Y))
    w = mweight(w)
    xmeans = colmean(zX, w)
    ymeans = colmean(zY, w)
    xstds = colstd(zX, w)
    ystds = colstd(zY, w)
    center!(zX, xmeans)
    center!(zY, ymeans)
    scale!(zX, xstds)
    scale!(zY, ystds)
    zX' * Diagonal(w) * zY
end

"""
    covm(X, w)
    covm(X, Y, w)
Compute covariance matrices.
* `X` : Data (n, p).
* `Y` : Data (n, q).
* `w` : Weights (n) of the observations.

`w` is internally normalized to sum to 1.

Uncorrected covariance matrix 
* of the columns of `X`: ==> (p, p) matrix 
* or between columns of `X` and `Y` : ==> (p, q) matrix.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
Y = rand(n, 3)
w = collect(1:n)

covm(X, w)
covm(X, Y, w)
```
"""
function covm(X, w)
    zX = copy(ensure_mat(X))
    w = mweight(w)
    xmeans = colmean(zX, w)
    center!(zX, xmeans)
    z = Diagonal(sqrt.(w)) * zX
    z' * z
end

function covm(X, Y, w)
    zX = copy(ensure_mat(X))
    zY = copy(ensure_mat(Y))
    w = mweight(w)
    xmeans = colmean(X, w)
    ymeans = colmean(Y, w)
    center!(zX, xmeans)
    center!(zY, ymeans)
    zX' * Diagonal(w) * zY
end

"""
    dummy(y)
Build a table of dummy variables from a categorical variable.
* `y` : A categorical variable.

## Examples
```julia
y = ["d", "a", "b", "c", "b", "c"]
#y =  rand(1:3, 7)
res = dummy(y)
pnames(res)
res.Y
```
"""
function dummy(y)
    ztab = tab(y)
    lev = ztab.keys
    nlev = length(lev)
    ni = collect(values(ztab))
    Y = BitArray(undef, length(y), nlev)
    for i = 1:nlev
        Y[:, i] = y .== lev[i]
    end
    Y = Float64.(Y)
    (Y = Y, lev = lev, ni = ni)
end

function dummy2(y)
    z = tab(y)
    lev = z.keys
    nlev = length(lev)
    ni = collect(values(tab(y)))
    res = list(nlev, BitVector)
    for i = 1:nlev
        res[i] = y .== lev[i]
    end
    Y = reduce(hcat, res)
    (Y = Y, lev = lev, ni = ni)
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
ensure_mat(X::AbstractVector) = Matrix(reshape(X, :, 1))
ensure_mat(X::Number) = reshape([X], 1, 1)
ensure_mat(X::LinearAlgebra.Adjoint) = Matrix(X)
ensure_mat(X::DataFrame) = Matrix(X)

"""
    findmax_cla(x, weights = nothing)
Find the most occurent level in `x`.
* `x` : A categorical variable.

If ex-aequos, the function returns the first.

## Examples
```julia
x = rand(1:3, 10)
tab(x)
findmax_cla(x)
```
"""
function findmax_cla(x, weights = nothing)
    isnothing(weights) ? weights = ones(length(x)) : nothing
    res = aggstat(weights; group = x, fun = sum)
    res.lev[argmax(res.X)]   # if equal, argmax takes the first
end

""" 
    fnorm(X)
    fnorm(X, w)
Frobenius norm of a matrix.
* `X` : A matrix (n, p).
* `w` : Weights (n) of the observations.

`w` is internally normalized to sum to 1.

The Frobenius norm of `X` is:
* sqrt(tr(X' * X)).

The weighted norm is:
* sqrt(tr(X' * D * X)), where D is the diagonal matrix of vector `w`.
"""
function fnorm(X)
    LinearAlgebra.norm(X)
end

function fnorm(X, w)
    # 1
    #sqrtD = Diagonal(sqrt.(mweight(w)))
    #sqrt(ssq(sqrtD * X))
    # 2
    # sqrt(sum(colnorm(X, mweight(w)).^2))
    # Faster:
    sqrt(sum(mweight(w)' * (X.^2)))
end

# Test: fnorm_2(X, w) = 

"""
    head(X)
Display the first rows of a dataset.

## Examples
```julia
X = rand(100, 5)
head(X)
```
"""
function head(X)
    display(X[1:3, :])
    if nro(X) > 3
        println("...")
    end
    psize(X)
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
iqr(x) = quantile(x, .75) - quantile(x, .25)


"""
    list(n::Integer)
Create a Vector{Any}(nothing, n).

`isnothing(object, i)` can be used to check if cell i is empty.

## Examples
```julia
list(5)
```
"""  
list(n::Integer) = Vector{Any}(nothing, n) 

"""
    list(n::Integer, type)
Create a Vector{type}(undef, n).

`isassigned(object, i)` can be used to check if cell i is empty.

## Examples
```julia
list(5, Float64)
list(5, Array{Float64})
list(5, Matrix{Float64})
```
"""  
list(n::Integer, type) = Vector{type}(undef, n)

""" 
    mad(x)
Compute the median absolute deviation (MAD),
adjusted by a factor (1.4826) for asymptotically normal consistency. 

## Examples
```julia
x = rand(100)
mad(x)
```
"""
mad(x) = 1.4826 * median(abs.(x .- median(x)))

""" 
    mweight(w)
Return a vector of weights that sums to 1.

## Examples
```julia
x = rand(10)
w = mweight(x)
sum(w)
```
"""
function mweight(w)
    zw = copy(vec(Float64.(w))) 
    mweight!(zw)
    zw
end

function mweight!(w::Vector{Float64})
    w ./= sum(w)
end

""" 
    nco(X)
Return the nb. columns of `X`.
"""
nco(X) = size(X, 2)

""" 
    normw(x, w)
Return the squared weighted norm of a vector.
* `x` : A vector (n).
* `w` : Weights (n) of the observations.

`w` is internally normalized to sum to 1.

The squared weighted norm of vector `x` is:
* x' * D * x, where D is the diagonal matrix of vector `w`.
"""
function normw(x, w::Vector)
    sqrt(dot(x, mweight(w) .* x))
end

""" 
    nro(X)
Return the nb. rows of `X`.
"""
nro(X) = size(X, 1)

""" 
    pnames(x)
Return the names of the elements of `x`.
"""
pnames(x) = propertynames(x)

""" 
    psize(x)
Return the type and size of `x`.
"""
function  psize(x)
    println(typeof(x))
    println(size(x))
end

"""
    recodcat2int(x; start = 1)
Recode a categorical variable to a integer variable
* `x` : Variable to recode.
* `start` : Integer value that will be set to the first category.

The numeric codes returned by the function are `Int64` and 
correspond to the sorted categories of `x`.

## Examples
```julia
x = ["b", "a", "b"]   
[x recodcat2int(x)]
recodcat2int(x; start = 0)
recodcat2int([25, 1, 25])
```
"""
function recodcat2int(x; start::Int64 = 1)
    z = dummy(x).Y
    ncla = nco(z)
    u = z .* collect(start:(start + ncla - 1))'
    u = sum(u; dims = 2)  ;
    Int64.(vec(u))
end

"""
    recodnum2cla(x, q)
Recode a continuous variable to classes.
* `x` : Variable to recode.
* `q` : Values separating the classes. 

## Examples
```julia
using Statistics
x = [collect(1:10); 8.1 ; 3.1] 
q = [3; 8]
zx = recodnum2cla(x, q)  
[x zx]
probs = [.33; .66]
q = quantile(x, probs) 
zx = recodnum2cla(x, q)  
[x zx]
```
"""
function recodnum2cla(x, q)
    zx = similar(x)
    q = sort(q)
    for i in eachindex(x)
        k = 1
        for j in eachindex(q)
            x[i] > q[j] ? k = k + 1 : nothing
        end
        zx[i] = k
    end
    Int64.(zx)
end

"""
    replacebylev(x, lev)
Replace the elements of a vector by levels of corresponding order.
* `x` : Vector (n) of values to replace.
* `lev` : Vector (nlev) containing the levels.

*Warning*: `x` and `lev` must contain the same number (nlev) of levels.

The ith sorted level in `x` is replaced by the ith sorted level of `lev`.

## Examples
```julia
x = [10; 4; 3; 3; 4; 4]
lev = ["B"; "C"; "AA"]
sort(lev)
[x replacebylev(x, lev)]
zx = string.(x)
[zx replacebylev(zx, lev)]

lev = [3; 0; -1]
[x replacebylev(x, lev)]
```
"""
function replacebylev(x, lev)
    n = length(x)
    lev = sort(lev)
    nlev = length(lev)
    @assert nlev == length(lev) "x and lev must contain the same number of levels."
    x_lev = tab(x).keys
    v = similar(lev, n)
    @inbounds for i = 1:nlev
        s = findall(x .== x_lev[i])
        v[s] .= lev[i] 
    end
    v
end

"""
    replacebylev2(x::Union{Int64, Array{Int64}}, lev::Array)
Replace the elements of an index-vector by levels.
* `x` : Vector (n) of values to replace.
* `lev` : Vector (nlev) containing the levels.

*Warning*: Let us note "nlev" the number of levels in `lev`. 
Vector `x` must contain integer values between 1 and nlev. 

Each element `x[i]` (i = 1,...,n) is replaced by `lev[x[i]]`.

## Examples
```julia
x = [2; 1; 2]
lev = ["B"; "C"; "AA"]
sort(lev)
[x replacebylev2(x, lev)]
replacebylev2([2], lev)
replacebylev2(2, lev)

x = [2; 1; 2]
lev = [3; 0; -1]
replacebylev2(x, lev)
```
"""
function replacebylev2(x::Union{Int64, Array{Int64}}, lev::Array)
    isa(x, Int64) ? x = [x] : x = vec(x)
    lev = vec(sort(lev))
    n = length(x)
    v = similar(lev, n)
    for i = 1:n
        v[i] = lev[x[i]]
    end
    v
end

"""
    replacedict(x, dict)
Replace the elements of a vector by levels defined in a dictionary.
* `x` : Vector (n) of values to replace.
* `dict` : A dictionary of the correpondances betwwen the old and new values.

## Examples
```julia
dict = Dict("a" => 1000, "b" => 1, "c" => 2)

x = ["c"; "c"; "a"; "a"; "a"]
replacedict(x, dict)

x = ["c"; "c"; "a"; "a"; "a"; "e"]
replacedict(x, dict)
```
"""
function replacedict(x, dict)
    replace(x, dict...)
end


"""
    rmcol(X, s)
Remove the columns of a matrix or the components of a vector 
having indexes `s`.
* `X` : Matrix or vector.
* `s` : Vector of the indexes.

## Examples
```julia
X = rand(5, 3) 
rmcol(X, [1, 3])
```
"""
function rmcol(X::Union{AbstractMatrix, DataFrame}, s::Union{Vector, BitVector, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[:, setdiff(1:end, Int64.(s))]
end

function rmcol(X::Vector, s::Union{Vector, BitVector, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[setdiff(1:end, Int64.(s))]
end

"""
    rmrow(X, s)
Remove the rows of a matrix or the components of a vector 
having indexes `s`.
* `X` : Matrix or vector.
* `s` : Vector of the indexes.

## Examples
```julia
X = rand(5, 2) 
rmrow(X, [1, 4])
```
"""
function rmrow(X::Union{AbstractMatrix, DataFrame}, s::Union{Vector, BitVector, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[setdiff(1:end, Int64.(s)), :]
end

function rmrow(X::Vector, s::Union{Vector, BitVector, Number})
    isa(s, BitVector) ? s = findall(s .== 1) : nothing
    X[setdiff(1:end, Int64.(s))]
end

"""
    rowmean(X)
Compute the mean of each row of `X`.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
rowmean(X)
```
""" 
rowmean(X) = vec(Statistics.mean(ensure_mat(X); dims = 2))

"""
    rowstd(X)
Compute the (uncorrected) standard deviation of each row of `X`.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
n, p = 5, 6
X = rand(n, p)
rowstd(X)
```
""" 
rowstd(X) = vec(Statistics.std(ensure_mat(X); dims = 2, corrected = false))

"""
    rowsum(X)
Compute the sum of each row of `X`.
* `X` : Data (n, p).

Return a vector.

## Examples
```julia
X = rand(5, 2) 
rowsum(X)
```
""" 
rowsum(X) = vec(sum(ensure_mat(X); dims = 2))

"""
    scale(X, v)
    scale!(X, v)
Scale each column of `X`.
* `X` : Data.
* `v` : Scaling factors.

## Examples
```julia
X = rand(5, 2) 
scale(X, colstd(X))
```
""" 
function scale(X, v)
    zX = copy(ensure_mat(X))
    scale!(zX, v)
    zX
end

function scale!(X::Matrix, v)
    p = nco(X)
    @inbounds for j = 1:p
        X[:, j] .= vcol(X, j) ./ v[j]
    end
end

# Below: Much slower and requires more memories
scale2(X, v) = mapslices(function f(x) ; x ./ v ; end, ensure_mat(X), dims = 2)

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
    summ(X, group; digits = 1)
Summarize a dataset (or a variable).
* `group` : A vector (n,) defing the groups.
* `digits` : Nb. digits in the outputs.

## Examples
```julia
X = rand(10, 3) 
res = summ(X)
pnames(res)
summ(X[:, 2]).res
```
"""
function summ(X; digits = 3)
    X = ensure_df(X)
    res = describe(X, :mean, :min, :max, :nmissing) ;
    insertcols!(res, 5, :n => nro(X) .- res.nmissing)
    for j = 2:4
        z = vcol(res, j)
        s = findall(isa.(z, Float64))
        res[s, j] .= round.(res[s, j], digits = digits)
        end
    (res = res, ntot = nro(X))
end

function summ(X, group; digits = 1)
    zgroup = sort(unique(group))
    for i in eachindex(zgroup)
        u = findall(group .== zgroup[i])
        z = X[u, :]
        res = summ(z; digits = digits).res
        println("Group: ", zgroup[i])
        println(res)
        println("") ; println("") 
        #println(repeat("-", 70))
    end
end

"""
    tab(x)
Univariate tabulation.
* `x` : Categorical variable.

In the output, the levels in `x` are sorted.
Levels and values can be get by `tab(x).keys` and `tab(x).vals`.

## Examples
```julia
x = rand(1:3, 10) 
tab(x)
```
"""
function tab(x)
    x = vec(x)
    sort(StatsBase.countmap(x))
end

"""
    tabnum(x)
Univariate tabulation (only integer classes).
* `x` : Categorical variable.

In the output, the levels in `x` are sorted.

## Examples
```julia
x = rand(1:3, 10) 
tabnum(x)
```
"""
function tabnum(x)
    x = vec(x)
    lev = sort(unique(x))
    cnt = StatsBase.counts(x)
    cnt = cnt[cnt .> 0]
    (cnt = cnt, lev = lev)
end

"""
    vcatdf(dat; cols = :intersect) 
    Vertical concatenation of a list of dataframes.
* `dat` : List of dataframes.
* `cols` : Determines the columns of the returned data frame.
    See ?DataFrames.vcat.

## Examples
```julia
using DataFrames
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
    vrow(X::Matrix, i)
    vrow(X::DataFrame, i)
    vrow(x::Vector, i)
    vcat(X::Matrix, j)
    vcat(X::DataFrame, j)
    vcat(x::Vector, j)
View of the i-th row(s) or j-th column(s) of a matrix `X`,
or of the i-th element(s) of vector `x`.
""" 
vrow(X, i) = view(X, i, :) 
vrow(X::DataFrame, i) = view(Matrix(X), i, :)
vrow(x::Vector, i) = view(x, i)

vcol(X, j) = view(X, :, j)
vcol(x::Vector, i) = view(x, i)
vcol(X::DataFrame, j) = view(Matrix(X), :, j)


