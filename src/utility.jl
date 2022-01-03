"""
    aggstat(X::Union{AbstractMatrix, AbstractVector}; group, fun = mean)
Compute the mean (or other statistic) of each column of `X`, by group.
* `X` : A matrix (n, p) or vector (n).
* `group` : A vector (n) defining the groups.
* `fun` : Function to compute.
""" 
function aggstat(X::Union{AbstractMatrix, AbstractVector}; group, 
        fun = mean)
    X = ensure_mat(X)
    group = vec(group)
    q = size(X, 2)
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

"""
    aggstat(X::DataFrame; var_nam, group_nam, fun = mean)
Compute the mean (or other statistic) of selected columns of `X`, by group.
* `X` : A dataframe.
* `var_nam` : Names (vector) of the variables to summarize.
* `group_nam` : Names (vector) of the group variables to consider.
* `fun` : Function given the statistic to compute.

Variables defined in `var_nam` and `group_nam` must be columns of `X`.
""" 
function aggstat(X::DataFrame; var_nam, group_nam, fun = mean)
    gdf = groupby(X, group_nam) 
    combine(gdf, var_nam .=> fun, renamecols = false)
end

"""
    center(X, v) 
Center each column of `X`.
* `X` : Data.
* `v` : Centering factors.
""" 
function center(X, v)
    M = copy(X)
    center!(M, v)
    M
end

function center!(X, v)
    p = size(X, 2)
    @inbounds for j = 1:p
        X[:, j] .= vcol(X, j) .- v[j]
    end
end

"""
    checkdupl(X; digits = 3)
Finding replicated rows in a dataset.
* `X` : A dataset.
* `digits` : Nb. digits used to round `X`.

## Examples
```julia
X = rand(5, 3)
Z = vcat(X, X[1:3, :])
checkdupl(X)
checkdupl(Z)

M = hcat(X, fill(missing, 5))
Z = vcat(M, M[1:3, :])
checkdupl(M)
checkdupl(Z)
```
"""
function checkdupl(X; digits = 3)
    X = round.(ensure_mat(X), digits = digits)
    n = size(X, 1)
    rownum1 = []
    rownum2 = []
    @inbounds for i = 1:n
        @inbounds for j = 1:n
            res = isequal(vrow(X, i), vrow(X, j))
            if res
                push!(rownum1, i)
                push!(rownum2, j)
            end
        end
    end
    u = findall(rownum1 .!= rownum2)
    res = DataFrame((rownum1 = rownum1[u], rownum2 = rownum2[u]))
    k = length(u)
    if k > 0
        res = res[1:Int64(k / 2), :]
    end
    res
end

"""
    checkmiss(X)
Finding rows with missing data in a dataset.
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
    colmeans(X)
    colmeans(X, w)
Compute the mean of each column of `X`.
* `X` : Data.
* `w` : Weights of the observations.

Return a vector.

For a true weighted mean, `w` must preliminary be normalized to sum to 1.
""" 
colmeans(X) = vec(Statistics.mean(X; dims = 1))
colmeans(X, w) = vec(w' * ensure_mat(X))

"""
    colvars(X)
    colvars(X, w)
Compute the (uncorrected) variance of each column of `X`.
* `X` : Data.
* `w` : Weights of the observations.

Return a vector.

**Note:** For a true weighted variance, `w` must preliminary be normalized to sum to 1.
""" 
colvars(X) = vec(Statistics.var(X; corrected = false, dims = 1))
function colvars(X, w)
    p = size(X, 2)
    z = colmeans(X, w)
    @inbounds for j = 1:p
        z[j] = dot(view(w, :), (vcol(X, j) .- z[j]).^2)        
    end
    z 
end
colstds(X) = sqrt.(colvars(X))
colstds(X, w) = sqrt.(colvars(X, w))


"""
    dummy(y)
Examples
≡≡≡≡≡≡≡≡≡≡
y = ["d", "a", "b", "c", "b", "c"]
#y =  rand(1:3, 7)
dummy(y)
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
ensure_mat(X::AbstractVector) = reshape(X, :, 1)
ensure_mat(X::Number) = reshape([X], 1, 1)
ensure_mat(X::LinearAlgebra.Adjoint) = Matrix(X)
ensure_mat(X::DataFrame) = Matrix(X)

"""
    findmax_cla(x, weights = nothing)
Find the most occurent level in `x`.
"""
function findmax_cla(x, weights = nothing)
    isnothing(weights) ? weights = ones(length(x)) : nothing
    res = aggstat(weights; group = x, fun = sum)
    res.lev[argmax(res.res)]   # if equal, argmax takes the first
end


"""
    iqr(x)
Compute the interquartile interval (IQR).
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
"""
mad(x) = 1.4826 * median(abs.(x .- median(x)))

"""
    matcov(X)
    matcov(X, w)
Compute the (uncorrected) covariance matrix of `X`.
* `X` : Data.
* `w` : Weights of the observations.

Uncorrected covariance matrix of the columns of `X`.

**Note:** For true weighted covariances, `w` must preliminary be normalized to sum to 1.
"""
matcov(X) = Statistics.cov(ensure_mat(X); corrected = false)

function matcov(X, w)
    X = ensure_mat(X)
    xmeans = colmeans(X, w)
    X = center(X, xmeans)
    z = Diagonal(sqrt.(w)) * X
    z' * z
end

""" 
    mweights(w)
Return a vector of weights that sums to 1.
"""
mweights(w) = w / sum(w)

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
    recodcat2num(x; start = 1)
Recode a categorical variable to a numeric variable
* `x` : Variable to recode.
* `start` : Numeric value that will be set to the first category.

The codes correspond to the sorted categories.

## Examples
```julia
x = ["b", "a", "b"] 
zx = recodcat2num(x)  
[x zx]
recodcat2num(x; start = 0)
recodcat2num([25, 1, 25])
```
"""
function recodcat2num(x; start = 1)
    z = dummy(x).Y
    ncla = size(z, 2)
    u = z .* collect(start:(start + ncla - 1))'
    u = sum(u; dims = 2)  ;
    u = vec(u)
end


"""
    recodnum2cla(x, q)
Recode a continuous variable to classes
* `x` : Variable to recode.
* `q` : Values separating the classes. 
## Examples
```julia
x = [collect(1:10); 8.1 ; 3.1] 
q = [3; 8] ;
zx = recodnum2cla(x, q) 
[x zx]
probs = [.33; .66] 
q = Statistics.quantile(x, probs) 
zx = recodnum2cla(x, q) ;
[x zx]
```
"""
function recodnum2cla(x, q)
    zx = similar(x)
    q = sort(q)
    for i = 1:length(x)
        k = 1
        for j = 1:length(q)
            x[i] > q[j] ? k = k + 1 : nothing
        end
        zx[i] = k
    end
    zx
end

"""
    replacebylev(x, lev)
Replaces the elements of x
by levels of corresponding order.

* `x` : Vector of values to replace.
* `lev` : Vector containing the levels.

The number of levels in `x` and `lev` must be the same.
Before replacement, the levels in `x` and `lev` are sorted.

## Examples
```julia
x = [10, 4, 3, 3, 4, 4]
lev = ["B"; "C"; "AA"]
[x replacebylev(x, lev)]
[string.(x) replacebylev(string.(x), lev)]

x = [10, 4, 3, 3, 4, 4]
lev = [3; 0; -1]
[x replacebylev(x, lev)]
```
"""
function replacebylev(x, lev)
    m = length(x)
    lev = sort(lev)
    nlev = length(lev)
    x_lev = tab(x).keys
    v = similar(lev, m)
    @inbounds for i = 1:nlev
        u = findall(x .== x_lev[i])
        v[u] .= lev[i] 
    end
    v
end

"""
    rmcol(X, s)
Remove the columns of `X` having indexes `s`.
## Examples
```julia
X = rand(5, 3) ; 
rmcol(X, 1:2)
rmcol(X, [1, 3])
```
"""
rmcol(X::AbstractMatrix, s::Union{Vector, Number}) = X[:, setdiff(1:end, Int64.(s))]
rmcol(x::Vector, s::Union{Vector, Number}) = x[setdiff(1:end, Int64.(s))]

"""
    rmrow(X, s)
Remove the rows of `X` having indexes `s`.
## Examples
```julia
X = rand(5, 2) ; 
rmrow(X, 2:3)
rmrow(X, [1, 4])
```
"""
rmrow(X::AbstractMatrix, s::Union{Vector, Number}) = X[setdiff(1:end, Int64.(s)), :]
rmrow(x::Vector, s::Union{Vector, Number}) = x[setdiff(1:end, Int64.(s))]

"""
    scale(X, v)
Scale each column of `X`.
* `X` : Data.
* `v` : Scaling factors.
""" 
function scale(X, v)
    M = copy(X)
    scale!(M, v)
    M
end

function scale!(X, v)
    p = size(X, 2)
    @inbounds for j = 1:p
        X[:, j] .= vcol(X, j) ./ v[j]
    end
end

# Below: Much slower and requires more memories
scale2(X, v) = mapslices(function f(x) ; x ./ v ; end, X, dims = 2)

"""
    sourcedir(path)
Include all the files contained in a directory.
"""
function sourcedir(path)
    z = readdir(path)  ## List of files in path
    n = length(z)
    for i in 1:n
        include(string(path, "/", z[i]))
    end
end

"""
    summ(X; digits = 3)
Summarize a dataset (or a variable).
"""
function summ(X; digits = 3)
    X = ensure_df(X)
    res = DataFrames.describe(X, :mean, :min, :max, :nmissing) ;
    insertcols!(res, 5, :n => size(X, 1) .- res.nmissing)
    for j = 2:4
        z = vcol(res, j)
        s = findall(isa.(z, Float64))
        res[s, j] .= round.(res[s, j], digits = digits)
        end
    (res = res, ntot = size(X, 1))
end

"""
    summ(X, group; digits = 1)
Summarize a dataset (or a variable), by group.
* `X` : Dataset (n, p) or (n,).
* `group` : A vector (n,) defing the groups.
"""
function summ(X, group; digits = 1)
    zgroup = sort(unique(group))
    for i = 1:length(zgroup)
        u = findall(group .== zgroup[i])
        z = X[u, :]
        res = summ(z).res
        println("Group: ", zgroup[i])
        println(res)
        println("") ; println("") 
        #println(repeat("-", 70))
    end
end

"""
    tab(x)
Univariate tabulation.
* `x` : Univariate class membership.

In the output, the levels in `x` are sorted.
Levels and values can be get by `tab(x).keys` and `tab(x).vals`.
"""
function tab(x)
    x = vec(x)
    sort(StatsBase.countmap(x))
end

"""
    tabnum(x)
Univariate tabulation (only integer classes).
* `x` : Univariate class membership.

In the output, the levels in `x` are sorted.
"""
function tabnum(x)
    x = vec(x)
    lev = sort(unique(x))
    cnt = StatsBase.counts(x)
    cnt = cnt[cnt .> 0]
    (cnt = cnt, lev = lev)
end

"""
    vrow(X::Matrix, j)
    vrow(x::Vector, i)
    vcol(X::Matrix, j)
View of the i-th row(s) or j-th column(s) of a matrix `X`,
or of the i-th element(s) of vector `x`.
""" 
vrow(X, i) = view(X, i, :)
vrow(x::Vector, i) = view(x, i)
vcol(X, j) = view(X, :, j)
vcol(x::Vector, i) = view(x, i)




