"""
    colmeans(X)
    colmeans(X, w)
Compute the mean of each column of `X`.
* `X` : Data.
* `w` : Weights of the observations.

Return a vector.

For a true mean, `w` must preliminary be normalized to sum to 1.
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

**Note:** For a true variance, `w` must preliminary be normalized to sum to 1.
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
    dummy(y)
Examples
≡≡≡≡≡≡≡≡≡≡
y = ["d", "a", "b", "c", "b", "c"]
#y =  rand(1:3, 7)
dummy(y)
"""
function dummy(y)
    z = tab(y)
    lev = z.keys
    nlev = length(lev)
    ni = collect(values(tab(y)))
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
    res = list(nlev)
    for i = 1:nlev
        res[i] = y .== lev[i]
    end
    Y = reduce(hcat, res)
    (Y = Y, lev = lev, ni = ni)
end

"""
    ensure_mat(X)
Reshape `X` to a matrix if necessary.
"""
ensure_mat(X::AbstractMatrix) = X
ensure_mat(X::AbstractVector) = reshape(X, :, 1)
ensure_mat(X::Number) = reshape([X], 1, 1)
ensure_mat(X::LinearAlgebra.Adjoint) = Matrix(X)

"""
    iqr(x)
Compute the interquartile interval (IQR).
"""
iqr(x) = quantile(x, .75) - quantile(x, .25)


"""
    list(n::Integer)
Create a Vector{Any}(undef, n).
"""  
list(n::Integer) = Vector{Any}(undef, n) 

""" 
    mad(x)
Compute the median absolute deviation (MAD),
adjusted by a factor (1.4826) for asymptotically normal consistency. 
"""
mad(x) = 1.4826 * median(abs.(x .- median(x)))

""" 
    mweights(w)
Return a vector of weights that sums to 1.
"""
mweights(w) = w / sum(w)

"""
    recod_cont2cla(x, q)
Recode a continuous variable to classes
* `x` : Variable to recode.
* `q` : Values separating the classes. 
## Examples
```julia
x = [collect(1:10); 8.1 ; 3.1] 
q = [3; 8] ;
zx = recod_cont2cla(x, q) 
[x zx]
```
"""
function recod2cla(x, q)
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
    rmrow(X, s)
Remove the rows of `X` having indexes `s`.
## Examples
```julia
X = rand(5, 2) ; 
rmrows(X, 2:3)
rmrows(X, [1, 4])
```
"""
rmrows(X::AbstractMatrix, s) = X[setdiff(1:end, s), :]
rmrows(X::AbstractVector, s) = X[setdiff(1:end, s)]

"""
    rmcols(X, s)
Remove the columns of `X` having indexes `s`.
## Examples
```julia
X = rand(5, 3) ; 
rmcols(X, 1:2)
rmcols(X, [1, 3])
```
"""
rmcols(X::AbstractMatrix, s) = X[:, setdiff(1:end, s)]

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

"""
    sourcedir(path)
Include all the files contained in a directory.
"""
function sourcedir(path)
    z = readdir(path)  ## List of files in path
    n = length(z)
    #sample(1:2, 1)
    for i in 1:n
        include(string(path, "/", z[i]))
    end
end

function tab(x)
    sort(StatsBase.countmap(x))
end

## tabnum(x) only works for the x classes represent integers
function tabnum(x)
    lev = sort(unique(x))
    cnt = StatsBase.counts(x)
    cnt = cnt[cnt .> 0]
    (cnt = cnt, lev = lev)
end

"""
    vrow(X, j)
    vcol(X, j)
View of the i-th row or j-th column of a matrix `X`.
""" 
vrow(X, i) = view(X, i, :)

vcol(X, j) = view(X, :, j)



