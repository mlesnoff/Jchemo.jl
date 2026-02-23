###### One vector 

""" 
    sumv(x)
    sumv(x, weights::ProbabilityWeights)
Compute the sum of a vector. 
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

## Examples
```julia
using Jchemo

n = 100
x = rand(n)
w = pweight(rand(n))

sumv(x)
sumv(x, w)
```
"""
sumv(x) = sum(x)

sumv(x, weights::ProbabilityWeights) = sum(x, weights::ProbabilityWeights)

""" 
    meanv(x)
    meanv(x, weights::ProbabilityWeights)
Compute the mean of a vector. 
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

## Examples
```julia
using Jchemo

n = 100
x = rand(n)
w = pweight(rand(n))

meanv(x)
meanv(x, w)
```
"""
meanv(x) = Statistics.mean(x)

meanv(x, weights::ProbabilityWeights) = sum(x, weights::ProbabilityWeights)

""" 
    normv(x)
    normv(x, weights::ProbabilityWeights)
Compute the norm of a vector.
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

The norm of vector `x` is computed by:
* sqrt(x' * x)

The weighted norm of vector `x` is computed by:
* sqrt(x' * D * x), where D is the diagonal matrix of vector `weights.values`.

## References

@gdkrmr,
https://discourse.julialang.org/t/julian-way-to-write-this-code/119348/17

@Stevengj, 
https://discourse.julialang.org/t/interesting-post-about-simd-dot-product-and-cosine-similarity/123282.

## Examples
```julia
using Jchemo

n = 1000
x = rand(n)
w = pweight(ones(n))

normv(x)
sqrt(n) * normv(x, w)
```
"""
normv(x) = sqrt(norm2v(x)) 

normv(x, weights::ProbabilityWeights) = sqrt(norm2v(x, weights))

""" 
    norm2v(x)
    norm2v(x, weights::ProbabilityWeights)
Compute the squared norm of a vector.
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

See function `normv`.

## Examples
```julia
using Jchemo

n = 1000
x = rand(n)
w = pweight(ones(n))

norm2v(x)
n * norm2v(x, w)
```
"""
norm2v(x) = dot(x, x) 

norm2v(x, weights::ProbabilityWeights) = sum(i -> x[i]^2 * weights.values[i], 1:length(x))

""" 
    stdv(x)
    stdv(x, weights::ProbabilityWeights)
Compute the uncorrected standard deviation of a vector.
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

## Examples
```julia
using Jchemo

n = 1000
x = rand(n)
w = pweight(rand(n))

stdv(x)
stdv(x, w)
```
"""
stdv(x) = Statistics.std(x; corrected = false) 

stdv(x, weights::Jchemo.ProbabilityWeights) = Statistics.std(x, weights; corrected = false)

""" 
    varv(x)
    varv(x, weights::ProbabilityWeights)
Compute the uncorrected variance of a vector.
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).

## Examples
```julia
using Jchemo

n = 1000
x = rand(n)
w = pweight(rand(n))

varv(x)
varv(x, w)
```
"""
varv(x) = Statistics.var(x; corrected = false) 

varv(x, weights::Jchemo.ProbabilityWeights) = Statistics.var(x, weights; corrected = false)

"""
    iqrv(x)
Compute the interquartile interval (IQR) of a vector.
* `x` : A vector (n).

## Examples
```julia
x = rand(100)
iqrv(x)
```
"""
iqrv(x) = quantile(x, .75) - quantile(x, .25)

""" 
    madv(x)

Compute the median absolute deviation (MAD) of a vector. 
* `x` : A vector (n).

This is the MAD adjusted by factor 1.4826 for asymptotically normal consistency.

## Examples
```julia
using Jchemo

x = rand(100)
madv(x)
```
"""
madv(x) = 1.4826 * median(abs.(x .- median(x)))

###### Two vectors

"""
    covv(x, y)
Compute uncorrected covariance between two vectors.
* `x` : vector (n).
* `y` : vector (n).

## Examples
```julia
using Jchemo

n = 5
x = rand(n)
y = rand(n)

covv(x, y)
```
"""
function covv(x, y)
    mux = meanv(x) 
    muy = meanv(y)
    n = length(x)
    sum(i -> (x[i] - mux) * (y[i] - muy), 1:n) / n
end 

function covv(x, y, weights::ProbabilityWeights)
    mux = meanv(x, weights) 
    muy = meanv(y, weights)
    sum(i -> (x[i] - mux) * (y[i] - muy) * weights.values[i], 1:length(x))
end 

"""
    cosv(x, y)
Compute cosinus between two vectors.
* `x` : vector (n).
* `y` : vector (n).

## References
@Stevengj, 
https://discourse.julialang.org/t/interesting-post-about-simd-dot-product-and-cosine-similarity/123282.

## Examples
```julia
using Jchemo

n = 5
x = rand(n)
y = rand(n)

cosv(x, y)
```
"""
function cosv(x, y)
    s = zero(x[begin]) * zero(y[begin])
    nx = ny = s
    @simd for i in eachindex(x, y)
        s = muladd(x[i], y[i], s)
        nx = muladd(x[i], x[i], nx)
        ny = muladd(y[i], y[i], ny)
    end
    s / sqrt(nx * ny)
end

function cosv(x, y, weights::Jchemo.ProbabilityWeights)
    s = zero(x[begin]) * zero(y[begin])
    nx = zero(x[begin]) * zero(y[begin])
    ny = zero(x[begin]) * zero(y[begin])
    @simd for i in eachindex(x, y)
        s = muladd(x[i] * weights.values[i], y[i], s)
        nx = muladd(x[i] * weights.values[i], x[i], nx)
        ny = muladd(y[i] * weights.values[i], y[i], ny)
    end
    s / sqrt(nx * ny)
end

"""
    corv(x, y)
Compute correlation between two vectors.
* `x` : vector (n).
* `y` : vector (n).

## References
@Stevengj, 
https://discourse.julialang.org/t/interesting-post-about-simd-dot-product-and-cosine-similarity/123282.

## Examples
```julia
using Jchemo

n = 5
x = rand(n)
y = rand(n)

corv(x, y)
```
"""
corv(x, y) = Statistics.cor(x, y)

function corv(x, y, weights::ProbabilityWeights)
    mux = meanv(x, weights) 
    muy = meanv(y, weights)
    sdx = stdv(x, weights)
    sdy = stdv(y, weights)
    s = zero(x[begin])
    @simd for i in eachindex(x)
        s = muladd((x[i] - mux) * (y[i] - muy),  weights.values[i], s)
    end
    s / (sdx * sdy)
end 

function corv_2(x, y)
    w = 1 / length(x)
    mux = meanv(x) 
    muy = meanv(y)
    sdx = stdv(x)
    sdy = stdv(y)
    s = zero(x[begin])
    @simd for i in eachindex(x)
        s = muladd((x[i] - mux) * (y[i] - muy),  w, s)
    end
    s / (sdx * sdy)
end 

function corv_3(x, y)
    cosv(fcenter(x, meanv(x)), fcenter(y, meanv(y)))
end 

###### Matrices

"""
    covm(X)
    covm(X, weights::ProbabilityWeights)
    covm(X, Y) 
    covm(X, Y, weights::ProbabilityWeights)
Compute a weighted covariance matrix.
* `X` : Data (n, p).
* `Y` : Data (n, q).
* `weights` : Weights (n) of the observations. Object of type `ProbabilityWeights` (e.g., generated by function `pweight`).

The function computes the uncorrected covariance matrix: 
* of the columns of `X`:  return a matrix (p, p),
* or between the columns of `X` and `Y` :  return a matrix (p, q).

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
Y = rand(n, 3)
w = pweight(rand(n))

covm(X, w)
covm(X, Y, w)
```
"""
covm(X) = Statistics.cov(X; corrected = false)

function covm(X, weights::ProbabilityWeights)
    zX = copy(ensure_mat(X))
    fcenter!(zX, colmean(zX, weights))
    rweight!(zX, sqrt.(weights.values))
    zX' * zX
end

covm(X, Y) = Statistics.cov(X, Y; corrected = false)

function covm(X, Y, weights::ProbabilityWeights)
    zX = copy(ensure_mat(X))
    zY = copy(ensure_mat(Y))
    fcenter!(zX, colmean(zX, weights))
    fcenter!(zY, colmean(zY, weights))
    zX' * rweight(zY, weights.values)
end

"""
    corm(X) 
    corm(X, Y) 
    corm(X, weights::ProbabilityWeights)
    corm(X, Y, weights::ProbabilityWeights)
Compute a weighted correlation matrix.
* `X` : Data (n, p).
* `Y` : Data (n, q).
* `weights` : Weights (n) of the observations. Object of type `ProbabilityWeights` (e.g., generated by function `pweight`).

Uncorrected correlation matrix 
* of the `X`-columns :  return a (p, p) matrix, 
* or between the `X`-columns and the `Y`-columns :  return a matrix (p, q).

## Examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
Y = rand(n, 3)
w = pweight(rand(n))

corm(X, w)
corm(X, Y, w)
```
"""
corm(X) = Statistics.cor(X)

function corm(X, weights::ProbabilityWeights)
    zX = copy(ensure_mat(X))
    fcenter!(zX, colmean(zX, weights))
    fscale!(zX, colstd(zX, weights))
    z = rweight(zX, sqrt.(weights.values))
    z' * z
end

corm(X, Y) = Statistics.cor(X, Y)

function corm(X, Y, weights::ProbabilityWeights)
    zX = copy(ensure_mat(X))
    zY = copy(ensure_mat(Y))
    fcenter!(zX, colmean(zX, weights))
    fcenter!(zY, colmean(zY, weights))
    fscale!(zX, colstd(zX, weights))
    fscale!(zY, colstd(zY, weights))
    zX' * rweight(zY, weights.values)
end

"""
    cosm(X)
    cosm(X, Y)
Compute a cosinus matrix.
* `X` : Data (n, p).
* `Y` : Data (n, q).

The function computes the cosinus matrix: 
* of the columns of `X`:  return a matrix (p, p),
* or between the columns of `X` and `Y` :  return a matrix (p, q).

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
    zX = fscale(X, colnorm(X))
    zY = fscale(Y, colnorm(Y))
    zX' * zY 
end

""" 
    frob(X)
    frob(X, weights::ProbabilityWeights)
    frob2(X)
    frob2(X, weights::ProbabilityWeights)
Frobenius norm of a matrix.
* `X` : A matrix (n, p).
* `weights` : Weights (n) of the observations. Object of type `ProbabilityWeights` (e.g., generated by function `pweight`).

The Frobenius norm of `X` is:
* sqrt(tr(X' * X)).

The weighted Frobenius norm is:
* sqrt(tr(X' * D * X)), where D is the diagonal matrix of vector `weights.values`.

Functions `frob2` are the squared versions of `frob`.

## References
@Stevengj, 
https://discourse.julialang.org/t/interesting-post-about-simd-dot-product-and-cosine-similarity/123282.
"""
frob(X) = sqrt(frob2(X))

frob(X, weights::ProbabilityWeights) = sqrt(frob2(X, weights)) 

function frob2(X)
    v = vec(ensure_mat(X))
    dot(v, v)
end

function frob2(X, weights::ProbabilityWeights) 
    n, p = size(X) 
    s = zero(X[begin])
    @inbounds for j = 1:p
        @simd for i = 1:n 
            s += weights.values[i] * X[i, j]^2
        end
    end
    s
end

