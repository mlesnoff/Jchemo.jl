
"""
    eposvd(D; nlv)
Pre-processing spectra by external parameter orthogonalization (EPO) (Roger et al 2003).
* `D` : Data (m, p) containing detrimental information.
* `nlv` : Nb. of first loadings vectors of D considered for the orthogonalization.

The objective is to remove from a dataset X (n, p) some "detrimental" 
information (e.g. humidity patterns in signals) that is defined by a dataset `D` (m, p).
The method orthogonalizes the observations (rows of X) to the 
"detrimental" sub-space, i.e. defined by the first `nlv` loadings vectors 
computed from a (non-centered) PCA of `D`.

Function `eposvd` makes a SVD factorization of `D` and returns 
two matrices:
* `M` (p, p) : The orthogonalization matrix that can be used to correct the X-data.
* `P` (p, `nlv`) : The matrix of the loading vectors of D. 

Any dataset Z can be corrected from the detrimental information `D` 
by computing Z_corrected = Z * `M`.

# References
Roger, J.-M., Chauchard, F., Bellon-Maurel, V., 2003. EPO-PLS external parameter 
orthogonalisation of PLS application to temperature-independent measurement 
of sugar content of intact fruits. 
Chemometrics and Intelligent Laboratory Systems 66, 191-204. 
https://doi.org/10.1016/S0169-7439(03)00051-0

Roger, J.-M., Boulet, J.-C., 2018. A review of orthogonal projections for calibration. 
Journal of Chemometrics 32, e3045. https://doi.org/10.1002/cem.3045

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "caltransfer.jld2") 
@load db dat
pnames(dat)
X1cal = dat.X1cal
X2cal = dat.X2cal
X1val = dat.X1val
X2val = dat.X2val

D = X1cal .- X2cal
nlv = 2
res = eposvd(D; nlv = nlv)
res.M      # orthogonalization matrix
res.P      # detrimental directions (columns of P = loadings of D)

# Corrected matrices

zX1 = X1val * res.M    
zX2 = X2val * res.M    

i = 1
f, ax = lines(zX1[i, :])
lines!(ax, zX2[i, :])
f
```
""" 
function eposvd(D; nlv)
    D = ensure_mat(D)
    m, p = size(D)
    nlv = min(nlv, m, p)
    Id = Diagonal(I, p)
    if nlv == 0 
        M = Id
        P = nothing
    else 
        P = svd(D).V[:, 1:nlv]
        M = Id - P * P'
    end
    (M = M, P = P)
end

"""
    detrend(X; pol = 1)
    detrend!(X::Matrix; pol = 1)
De-trend transformation of each row of a matrix X. 
* `X` : X-data.
* `pol` : Polynom order.

The function fits a polynomial regression to each observation
and returns the residuals.

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wl = names(dat.X)
wl_num = parse.(Float64, wl)

Xp = detrend(X)
plotsp(Xp[1:30, :], wl_num).f
```
""" 
function detrend(X; pol = 1)
    zX = copy(ensure_mat(X))
    Jchemo.detrend!(zX; pol = pol)
    zX
end
function detrend!(X::Matrix; pol = 1)
    n, p = size(X)
    vX = similar(X, p, pol + 1)
    for j = 0:pol
        vX[:, j + 1] .= collect(1:p).^j
    end
    vXt = vX'
    vXtvX = vXt * vX
    tol = sqrt(eps(real(float(one(eltype(vXtvX))))))
    A = pinv(vXtvX, rtol = tol) * vXt
    # Not faster: @Threads.threads
    @inbounds for i = 1:n
        y = vrow(X, i)
        X[i, :] .= y - vX * A * y
    end
end

"""
    fdif(X; f = 2)
    fdif!(M::Matrix, X::Matrix; f = 2)
Compute finite differences for each row of a matrix X. 
* `X` : X-data (n, p).
* `M` : Pre-allocated output matrix (n, p - f + 1).
* `f` : Size of the window (nb. points involved) for the finite differences.
    The range of the window (= nb. intervals of two successive colums) is f - 1.

The finite differences can be used for computing discrete derivates.
The method reduces the column-dimension: (n, p) --> (n, p - f + 1). 

The in-place function stores the output in `M`.

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wl = names(dat.X)
wl_num = parse.(Float64, wl)

Xp = fdif(X; f = 10)
plotsp(Xp[1:30, :]).f
```
""" 
function fdif(X; f = 2)
    X = ensure_mat(X)
    n, p = size(X)
    M = similar(X, n, p - f + 1)
    fdif!(M, X; f)
    M
end

function fdif!(M::Matrix, X::Matrix; f = 2)
    p = size(X, 2)
    zp = p - f + 1
    @Threads.threads for j = 1:zp
        M[:, j] .= vcol(X, j + f - 1) .- vcol(X, j)
    end
end

""" 
    interpl(X, wl; wlfin, fun = cubic_spline)
Sampling signals by interpolation.
* `X` : Matrix (n, p) of signals (rows).
* `wl` : Values representing the column "names" of `X`. 
    Must be a numeric vector of length p, or an AbstractRange.
* `wlfin` : Final values where to interpolate within the range of `wl`.
    Must be a numeric vector, or an AbstractRange.
* `fun` : Function defining the interpolation method.

The function uses package DataInterpolations.jl.

Possible values of `fun` (methods) are:
- `linear_int`: A linear interpolation (LinearInterpolation).
- `quadratic_int`: A quadratic interpolation (QuadraticInterpolation).
- `quadratic_spline`: A quadratic spline interpolation(QuadraticSpline).
- `cubic_spline`: A cubic spline interpolation (CubicSpline)

## References
Package Interpolations.jl
https://github.com/PumasAI/DataInterpolations.jl
https://htmlpreview.github.io/?https://github.com/PumasAI/DataInterpolations.jl/blob/v2.0.0/example/DataInterpolations.html

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
wl = names(X)
wl_num = parse.(Float64, wl) 

plotsp(X[1:10,:], wl_num).f

wlfin = collect(range(500, 2400, length = 10))
#wlfin = range(500, 2400, length = 10)
Xp = interpl(X[1:10, :], wl_num; wlfin = wlfin) 
plotsp(Xp, wlfin).f

Xp = interpl_mon(X[1:10, :], wl_num; wlfin = wlfin) ;
plotsp(Xp, wlfin).f
```
""" 
function interpl(X, wl; wlfin, fun = cubic_spline)
    X = ensure_mat(X)
    n = size(X, 1)
    q = length(wlfin)
    zX = similar(X, n, q)
    # Not faster: @Threads.threads
    @inbounds for i = 1:n
        itp = fun(vrow(X, i), wl)
        zX[i, :] .= itp.(wlfin)
    end
    zX
end
# Due to conflicts with Interpolations.jl
linear_int(y, x) = DataInterpolations.LinearInterpolation(y, x)
quadratic_int(y, x) = DataInterpolations.QuadraticInterpolation(y, x)
quadratic_spline(y, x) = DataInterpolations.QuadraticSpline(y, x)
cubic_spline(y, x) = DataInterpolations.CubicSpline(y, x)

""" 
    interpl_mon(X, wl; wlfin, fun = FritschCarlsonMonotonicInterpolation)
Sampling signals by monotonic interpolation.
* `X` : Matrix (n, p) of signals (rows).
* `wl` : Values representing the column "names" of `X`. 
    Must be a numeric vector of length p, or an AbstractRange.
* `wlfin` : Values where to interpolate within the range of `wl`.
    Must be a numeric vector, or an AbstractRange.
* `fun` : Function defining the interpolation method.

See e.g. https://en.wikipedia.org/wiki/Monotone_cubic_interpolation.

The function uses package Interpolations.jl.

Possible values of `fun` (methods) are:
- `LinearMonotonicInterpolation`
- `FiniteDifferenceMonotonicInterpolation` : Classic cubic
- `CardinalMonotonicInterpolation`
- `FritschCarlsonMonotonicInterpolation`
- `FritschButlandMonotonicInterpolation`
- `SteffenMonotonicInterpolation`
See https://github.com/JuliaMath/Interpolations.jl/pull/243/files#diff-92e3f2a374c9a54769084bad1bbfb4ff20ee50716accf008074cda7af1cd6149

See '?interpl' for examples. 

## References
Package Interpolations.jl
https://github.com/JuliaMath/Interpolations.jl

Fritsch & Carlson (1980), "Monotone Piecewise Cubic Interpolation", 
doi:10.1137/0717021.

Fritsch & Butland (1984), "A Method for Constructing Local Monotone Piecewise 
Cubic Interpolants", doi:10.1137/0905021.

Steffen (1990), "A Simple Method for Monotonic Interpolation 
in One Dimension", http://adsabs.harvard.edu/abs/1990A%26A...239..443S
""" 
function interpl_mon(X, wl; wlfin, fun = FritschCarlsonMonotonicInterpolation)
    X = ensure_mat(X)
    n = size(X, 1)
    q = length(wlfin)
    zX = similar(X, n, q)
    # Not faster: @Threads.threads
    @inbounds for i = 1:n
        itp = interpolate(wl, vrow(X, i), fun())
        zX[i, :] .= itp.(wlfin)
    end
    zX
end

"""
    mavg(X; f)
    mavg!(X::Matrix; f)
Moving averages smoothing of each row of X-data.
* `X` : X-data.
* `f` : Size (nb. points involved) of the filter.

The smoothing is computed by convolution (with padding), with function 
imfilter of package ImageFiltering.jl. The centered kernel is ones(`f`) / `f`.
Each returned point is located on the center of the kernel.

## References
Package ImageFiltering.jl
https://github.com/JuliaImages/ImageFiltering.jl

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wl = names(dat.X)
wl_num = parse.(Float64, wl)

Xp = mavg(X; f = 10) 
plotsp(Xp[1:30, :], wl_num).f
```
""" 
function mavg(X; f)
    zX = copy(ensure_mat(X))
    mavg!(zX; f)
    zX
end

function mavg!(X::Matrix; f)
    n, p = size(X)
    f = Int64(f)
    kern = ImageFiltering.centered(ones(f) / f) ;
    out = similar(X, p) 
    @inbounds for i = 1:n
        imfilter!(out, vrow(X, i), kern)
        X[i, :] .= out
    end
    # Not faster
    #@Threads.threads for i = 1:n
    #    X[i, :] .= imfilter(vrow(X, i), kern)
    #end
end

"""
    mavg_runmean(X, f)
    mavg_runmean!(M::Matrix, X::Matrix; f)
Moving average smoothing of each row of a matrix X.
* `X` : X-data (n, p).
* `M` : Pre-allocated output matrix (n, p - f + 1).
* `f` : Size (nb. points involved) of the filter.

The smoothing is computed by convolution, without padding (which reduces 
the column dimension). The function is an adaptation/simplification of function 
runmean (V. G. Gumennyy) of package Indicators.jl. See
https://github.com/dysonance/Indicators.jl/blob/a449c1d68487c3a8fea0008f7abb3e068552aa08/src/run.jl.
The kernel is ones(`f`) / `f`. Each returned point is located on the 1st unit of the kernel.
In general, this function can be faster than mavg, especialy for in-place versions.

The in-place function stores the output in `M`.

## References
Package Indicators.jl
https://github.com/dysonance/Indicators.jl

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wl = names(dat.X)
wl_num = parse.(Float64, wl)

Xp = mavg_runmean(X; f = 10) 
plotsp(Xp[1:30, :]).f
```
""" 
function mavg_runmean(X; f)
    X = ensure_mat(X)
    n, p = size(X)
    M = similar(X, n, p - f + 1)
    mavg_runmean!(M, X; f)
    M
end

function mavg_runmean!(M::Matrix, X::Matrix; f)
    n, zp = size(M)
    out = similar(M, zp)
    # Not faster: @Threads.threads 
    @inbounds for i = 1:n
        Jchemo.runmean!(out, vrow(X, i); f)
        M[i, :] .= out    
    end
end

## This is an adaptation from function runmean  (V. G. Gumennyy)
## of package Indicators.jl
function runmean!(out, x; f)
    ## x : (n,)
    ## out : (n - f + 1,)
    ## f : integer
    ## There is no padding
    ## Location on the 1st unit of the kernel
    n = length(x)
    zsum = 0.
    @inbounds for i = 1:f
        zsum += x[i]
    end
    out[1] = zsum / f
    @inbounds for i = (f + 1):n
        zsum += x[i] - x[i - f] 
        out[i - f + 1] = zsum / f
    end
end

""" 
    savgk(m, pol, d)
Compute the kernel of the Savitzky-Golay filter.
* `m` : Nb. points of the half window (m >= 1) 
    --> the size of the kernel is odd (f = 2 * m + 1): 
    x[-m], x[-m+1], ..., x[0], ...., x[m-1], x[m].
* `pol` : Polynom order (1 <= pol <= 2 * m).
    The case "pol = 0" (simple moving average) is not allowed by the funtion.
* `d` : Derivation order (0 <= d <= pol).
    If `d = 0`, there is no derivation (only polynomial smoothing).

## References
Luo, J., Ying, K., Bai, J., 2005. Savitzky–Golay smoothing and differentiation 
filter for even number data. Signal Processing 85, 1429–1434.
https://doi.org/10.1016/j.sigpro.2005.02.002

## Examples
```julia
res = savgk(21, 3, 2)
pnames(res)
res.S 
res.G 
res.kern
```
""" 
function savgk(m, pol, d)
    @assert m >= 1 "m must be >= 1"
    @assert pol >= 1 && pol <= 2 * m "pol must agree with: 1 <= pol <= 2 * m"
    @assert 0 <= d && d <= pol "d must agree with: 0 <= d <= pol"
    f = 2 * m + 1
    S = zeros(Int64(f), Int64(pol) + 1) ;
    u = collect(-m:m)
    @inbounds for j in 0:pol
        S[:, j + 1] .= u.^j
    end
    G = S * inv(S' * S)
    kern = factorial(d) * vcol(G, d + 1)
    (S = S, G = G, kern = kern)
end

"""
    savgol(X; f, pol, d)
    savgol!(X::Matrix; f, pol, d)
Savitzky-Golay smoothing of each row of a matrix `X`.
* `X` : X-data (n, p).
* `f` : Size of the filter (nb. points involved in the kernel). Must be odd and >= 3.
    The half-window size is m = (f - 1) / 2.
* `pol` : Polynom order (1 <= pol <= f - 1).
* `d` : Derivation order (0 <= d <= pol).

The smoothing is computed by convolution (with padding), with function 
imfilter of package ImageFiltering.jl. Each returned point is located on the center 
of the kernel. The kernel is computed with function `savgk`.

## References 
Luo, J., Ying, K., Bai, J., 2005. Savitzky–Golay smoothing and differentiation filter for 
even number data. Signal Processing 85, 1429–1434. https://doi.org/10.1016/j.sigpro.2005.02.002

Savitzky, A., Golay, M.J.E., 2002. Smoothing and Differentiation of Data by Simplified Least 
Squares Procedures. [WWW Document]. https://doi.org/10.1021/ac60214a047

Schafer, R.W., 2011. What Is a Savitzky-Golay Filter? [Lecture Notes]. 
IEEE Signal Processing Magazine 28, 111–117. https://doi.org/10.1109/MSP.2011.941097

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wl = names(dat.X)
wl_num = parse.(Float64, wl)

f = 21 ; pol = 3 ; d = 2 ; 
Xp = savgol(X; f = f, pol = pol, d = d) 
plotsp(Xp[1:30, :], wl_num).f
```
""" 
function savgol(X; f, pol, d)
    zX = ensure_mat(copy(X))
    savgol!(zX; f, pol, d)
    zX
end

function savgol!(X::Matrix; f, pol, d)
    X = ensure_mat(X)
    @assert isodd(f) && f >= 3 "f must be odd and >= 3"
    n, p = size(X)
    m = (f - 1) / 2
    kern = savgk(m, pol, d).kern
    zkern = ImageFiltering.centered(kern)
    out = similar(X, p)
    @inbounds for i = 1:n
        ## convolution with "replicate" padding
        imfilter!(out, vrow(X, i), reflect(zkern))
        X[i, :] .= out
    end
    # Not faster
    #@Threads.threads for i = 1:n
    #    X[i, :] .= imfilter(vrow(X, i), reflect(zkern))
    #end
end

"""
    snv(X; cent = true, scal = true)
    snv!(X::Matrix; cent = true, scal = true)
Standard-normal-variate (SNV) transformation of each row of X-data.
* `X` : X-data.
* `cent` : Logical indicating if the centering in done.
* `scal` : Logical indicating if the scaling in done.

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wl = names(dat.X)
wl_num = parse.(Float64, wl)

Xp = snv(X) 
plotsp(Xp[1:30, :], wl_num).f
```
""" 
function snv(X; cent = true, scal = true)
    zX = ensure_mat(copy(X))
    snv!(zX; cent = cent, scal = scal)
    zX
end

function snv!(X::Matrix; cent = true, scal = true) 
    n, p = size(X)
    cent ? mu = rowmean(X) : mu = zeros(n)
    scal ? s = rowstd(X) : s = ones(n)
    # Not faster: @Threads.threads
    @inbounds for j = 1:p
        X[:, j] .= (vcol(X, j) .- mu) ./ s
    end
end


