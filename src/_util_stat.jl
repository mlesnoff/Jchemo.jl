#### One vector 

""" 
    sumv(x)
    sumv(x, weights::Weight)
Compute the sum of a vector. 
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).

## Examples
```julia
using Jchemo

n = 100
x = rand(n)
w = mweight(rand(n)) 

sumv(x)
sumv(x, w)
```
"""
sumv(x) = Base.sum(x)

function sumv(x, weights::Weight)
    s = zero(x[begin])
    @simd for i in eachindex(x)
        s = muladd(x[i],  weights.w[i], s)
    end
    s
end

""" 
    meanv(x)
    meanv(x, weights::Weight)
Compute the mean of a vector. 
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).

## Examples
```julia
using Jchemo

n = 100
x = rand(n)
w = mweight(rand(n)) 

meanv(x)
meanv(x, w)
```
"""
meanv(x) = Statistics.mean(x)

meanv(x, weights::Weight) = sumv(x, weights::Weight)

""" 
    stdv(x)
    stdv(x, weights::Weight)
Compute the uncorrected standard deviation of a vector.
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).

## Examples
```julia
using Jchemo

n = 1000
x = rand(n)
w = mweight(rand(n))

stdv(x)
stdv(x, w)
```
"""
stdv(x) = Statistics.std(x; corrected = false) 

function stdv(x, weight::Weight)
    n = length(x)
    mu = meanv(x, weight)
    sqrt(sum(i -> (x[i] - mu)^2 * weight.w[i], 1:n))
end

""" 
    varv(x)
    varv(x, weights::Weight)
Compute the uncorrected variance of a vector.
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).

## Examples
```julia
using Jchemo

n = 1000
x = rand(n)
w = mweight(rand(n))

varv(x)
varv(x, w)
```
"""
varv(x) = Statistics.var(x; corrected = false) 

function varv(x, weight::Weight)
    n = length(x)
    mu = meanv(x, weight)
    sum(i -> (x[i] - mu)^2 * weight.w[i], 1:n)
end

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

This is the MAD adjusted by a factor (1.4826) for asymptotically 
normal consistency.

## Examples
```julia
using Jchemo

x = rand(100)
madv(x)
```
"""
madv(x) = 1.4826 * median(abs.(x .- median(x)))

""" 
    normv(x)
    normv(x, weights::Weight)
Compute the norm of a vector.
* `x` : A vector (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).

The norm of vector `x` is computed by:
* sqrt(x' * x)

The weighted norm of vector `x` is computed by:
* sqrt(x' * D * x), where D is the diagonal matrix of vector `weights.w`.

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
w = mweight(ones(n))

normv(x)
sqrt(n) * normv(x, w)
```
"""
normv(x) = sqrt(dot(x, x)) 

normv(x, weights::Weight) = sqrt(sum(i -> x[i]^2 * weights.w[i], 1:length(x)))

#### Two vectors

"""
    cosv(x, y)
Compute uncorrected covariance between two vectors.
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

covv(x, y)
```
"""
covv(x, y) = Statistics.cov(x, y; corrected = false)

function covv(x, y, weights::Weight)
    mux = meanv(x, weights) 
    muy = meanv(y, weights)
    s = zero(x[begin])
    @simd for i in eachindex(x)
        s = muladd((x[i] - mux) * (y[i] - muy),  weights.w[i], s)
    end
    s
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

function corv(x, y, weights::Weight)
    mux = meanv(x, weights) 
    muy = meanv(y, weights)
    sdx = stdv(x, weights)
    sdy = stdv(y, weights)
    s = zero(x[begin])
    @simd for i in eachindex(x)
        s = muladd((x[i] - mux) * (y[i] - muy),  weights.w[i], s)
    end
    s / (sdx * sdy)
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
cosv(x, y) = dot(x, y) / sqrt(dot(x, x) * dot(y, y))

function cosv(x, y, weights::Weight)
    w = weights.w
    zy = fweight(y, w)
    dot(x, zy) / sqrt(dot(x, fweight(x, w)) * dot(y, zy))
end

#### Matrices

"""
    covm(X)
    covm(X, weights::Weight)
    covm(X, Y) 
    covm(X, Y, weights::Weight)
Compute a weighted covariance matrix.
* `X` : Data (n, p).
* `Y` : Data (n, q).
* `weights` : Weights (n) of the observations. Object of type 
    `Weight` (e.g. generated by function `mweight`).

The function computes the uncorrected covariance matrix: 
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
covm(X) = Statistics.cov(X; corrected = false)

function covm(X, weights::Weight)
    zX = copy(ensure_mat(X))
    fcenter!(zX, colmean(zX, weights))
    fweight!(zX, sqrt.(weights.w))
    zX' * zX
end

covm(X, Y) = Statistics.cov(X, Y; corrected = false)

function covm(X, Y, weights::Weight)
    zX = copy(ensure_mat(X))
    zY = copy(ensure_mat(Y))
    fcenter!(zX, colmean(zX, weights))
    fcenter!(zY, colmean(zY, weights))
    zX' * fweight(zY, weights.w)
end

"""
    corm(X) 
    corm(X, Y) 
    corm(X, weights::Weight)
    corm(X, Y, weights::Weight)
Compute a weighted correlation matrix.
* `X` : Data (n, p).
* `Y` : Data (n, q).
* `weights` : Weights (n) of the observations. Object of type 
    `Weight` (e.g. generated by function `mweight`).

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
corm(X) = Statistics.cor(X)

function corm(X, weights::Weight)
    zX = copy(ensure_mat(X))
    fcenter!(zX, colmean(zX, weights))
    fscale!(zX, colstd(zX, weights))
    z = fweight(zX, sqrt.(weights.w))
    z' * z
end

corm(X, Y) = Statistics.cor(X, Y)

function corm(X, Y, weights::Weight)
    zX = copy(ensure_mat(X))
    zY = copy(ensure_mat(Y))
    fcenter!(zX, colmean(zX, weights))
    fcenter!(zY, colmean(zY, weights))
    fscale!(zX, colstd(zX, weights))
    fscale!(zY, colstd(zY, weights))
    zX' * fweight(zY, weights.w)
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
    zX = fscale(X, colnorm(X))
    zY = fscale(Y, colnorm(Y))
    zX' * zY 
end

""" 
    frob(X)
    frob(X, weights::Weight)
    frob2(X)
    frob2(X, weights::Weight)
Frobenius norm of a matrix.
* `X` : A matrix (n, p).
* `weights` : Weights (n) of the observations. Object of type 
    `Weight` (e.g. generated by function `mweight`).

The Frobenius norm of `X` is:
* sqrt(tr(X' * X)).

The Frobenius weighted norm is:
* sqrt(tr(X' * D * X)), where D is the diagonal matrix of vector `w`.

Functions `frob2` are the squared versions of `frob`.

## References
@Stevengj, 
https://discourse.julialang.org/t/interesting-post-about-simd-dot-product-and-cosine-similarity/123282.
"""
frob(X) = sqrt(frob2(X))

frob(X, weights::Weight) = sqrt(frob2(X, weights)) 

function frob2(X)
    v = vec(ensure_mat(X))
    dot(v, v)
end

function frob2(X, weights::Weight) 
    n, p = size(X) 
    s = zero(X[begin])
    @inbounds for j = 1:p
        @simd for i = 1:n 
            s += weights.w[i] * X[i, j]^2
        end
    end
    s
end

