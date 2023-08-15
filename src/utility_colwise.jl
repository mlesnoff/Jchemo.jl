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
    @inbounds for i = 1:p
        z[i] = mad(vcol(X, i))        
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
    #@inbounds for i = 1:p
    #    x = vcol(X, i)
    #    z[i] = sqrt(dot(x, w .* x))
    #end
    #z 
    # Faster:
    vec(sqrt.(mweight(w)' * X.^2))
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
colstd(X) = vec(Statistics.std(ensure_mat(X); corrected = false, dims = 1))
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
    @inbounds for i = 1:p
        z[i] = dot(w, (vcol(X, i) .- z[i]).^2)        
    end
    z 
end

####### SKIP MISSING

function colmeanskip(X)
    X = ensure_mat(X)
    [mean(skipmissing(vcol(X, i))) for i in 1:nco(X)]
end

function colstdskip(X)
    X = ensure_mat(X)
    [std(skipmissing(vcol(X, i)); corrected = false) for i in 1:nco(X)]
end

function colsumskip(X)
    X = ensure_mat(X)
    [sum(skipmissing(vcol(X, i))) for i in 1:nco(X)]
end

function colvarskip(X)
    X = ensure_mat(X)
    [var(skipmissing(vcol(X, i)); corrected = false) for i in 1:nco(X)]
end

## With weights
function colmeanskip(X, w)
    X = ensure_mat(X)
    p = nco(X)
    w = collect(w) # rmrow does not accept UnitRange
    z = zeros(p)
    for i = 1:p
        s = ismissing.(vcol(X, i))
        z[i] = sum(mweight(rmrow(w, s)) .* rmrow(X[:, i], s))
    end
    z
end

colsumskip(X, w) = colmeanskip(X, w)

colstdskip(X, w) = sqrt.(colvarskip(X, w))

function colvarskip(X, w)
    X = ensure_mat(X)
    p = nco(X)
    w = collect(w)
    z = colmeanskip(X, w)
    @inbounds for i = 1:p
        s = ismissing.(vcol(X, i))
        w = mweight(rmrow(w, s))
        z[i] = dot(w, (rmrow(X[:, i], s) .- z[i]).^2)        
    end
    z 
end

