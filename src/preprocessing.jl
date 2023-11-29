"""
    detrend(X; degree = 1)
    detrend!(X::Matrix; degree = 1)
De-trend transformation of each row of a matrix X. 
* `X` : X-data.
* `degree` : Polynom order.

The function fits a polynomial regression to each observation
and returns the residuals.

## Examples
```julia
using JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wlstr = names(dat.X)
wl = parse.(Float64, wlstr)

Xp = detrend(X)
plotsp(Xp[1:30, :], wl).f
```
""" 

function detrend(X; kwargs...)
    par = recovkwargs(Par, kwargs)
    Detrend(kwargs, par)
end

function transf(object::Detrend, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Detrend, X::Matrix)
    n, p = size(X)
    degree = object.par.degree
    vX = similar(X, p, degree + 1)
    for j = 0:degree
        vX[:, j + 1] .= collect(1:p).^j
    end
    vXt = vX'
    vXtvX = vXt * vX
    tol = sqrt(eps(real(float(one(eltype(vXtvX))))))
    A = pinv(vXtvX, rtol = tol) * vXt
    ## Not faster: @Threads.threads
    @inbounds for i = 1:n
        y = vrow(X, i)
        X[i, :] .= y - vX * A * y
    end
end

"""
    fdif(X; npoint = 2)
    fdif!(M::Matrix, X::Matrix; npoint = 2)
Compute finite differences for each row of a matrix X. 
* `X` : X-data (n, p).
* `M` : Pre-allocated output matrix (n, p - npoint + 1).
* `npoint` : Nb. points involved in the window for the finite differences.
    The range of the window (= nb. intervals of two successive colums) is npoint - 1.

The finite differences can be used for computing discrete derivates.
The method reduces the column-dimension: (n, p) --> (n, p - npoint + 1). 

The in-place function stores the output in `M`.

## Examples
```julia
using JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wlstr = names(dat.X)
wl = parse.(Float64, wlstr)

Xp = fdif(X; npoint = 10)
plotsp(Xp[1:30, :]).f
```
""" 
function fdif(X; kwargs...)
    par = recovkwargs(Par, kwargs)
    Fdif(kwargs, par)
end

function transf(object::Fdif, X)
    X = ensure_mat(X)
    n, p = size(X)
    npoint = object.par.npoint
    M = similar(X, n, p - npoint + 1)
    transf!(object, X, M)
    M
end

function transf!(object::Fdif, X::Matrix, M::Matrix)
    p = nco(X)
    npoint = object.par.npoint
    zp = p - npoint + 1
    @Threads.threads for j = 1:zp
        M[:, j] .= vcol(X, j + npoint - 1) .- vcol(X, j)
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
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
wlstr = names(X)
wl = parse.(Float64, wlstr) 

plotsp(X[1:10,:], wl).f

wlfin = collect(range(500, 2400, length = 10))
#wlfin = range(500, 2400, length = 10)
Xp = interpl(X[1:10, :], wl; wlfin = wlfin) 
plotsp(Xp, wlfin).f

Xp = interpl_mon(X[1:10, :], wl; wlfin = wlfin) ;
plotsp(Xp, wlfin).f
```
"""
function interpl(X; kwargs...)
    par = recovkwargs(Par, kwargs)
    Interpl(kwargs, par)
end

function transf(object::Interpl, X)
    X = ensure_mat(X)
    n = nro(X)
    p = length(object.par.wlfin)
    M = similar(X, n, p)
    transf!(object, X, M)
    M
end

function transf!(object::Interpl, X::Matrix, M::Matrix)
    n = nro(X)
    wl = object.par.wl 
    wlfin = object.par.wlfin 
    fun = DataInterpolations.CubicSpline
    ## Not faster: @Threads.threads
    @inbounds for i = 1:n
        itp = fun(vrow(X, i), wl)
        M[i, :] .= itp.(wlfin)
    end
end
#cubic_spline(y, x) = DataInterpolations.CubicSpline(y, x)
#linear_int(y, x) = DataInterpolations.LinearInterpolation(y, x)
#quadratic_int(y, x) = DataInterpolations.QuadraticInterpolation(y, x)
#quadratic_spline(y, x) = DataInterpolations.QuadraticSpline(y, x)

"""
    mavg(X; npoint)
    mavg!(X::Matrix; npoint)
Moving averages smoothing of each row of X-data.
* `X` : X-data.
* `npoint` : Nb. points involved in the window 

The smoothing is computed by convolution (with padding), 
with function imfilter of package ImageFiltering.jl. The centered 
kernel is ones(`npoint`) / `npoint`. Each returned point is located on 
the fcenter of the kernel.

## References
Package ImageFiltering.jl
https://github.com/JuliaImages/ImageFiltering.jl

## Examples
```julia
using JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wlstr = names(dat.X)
wl = parse.(Float64, wlstr)

Xp = mavg(X; npoint = 10) 
plotsp(Xp[1:30, :], wl).f
```
""" 
function mavg(X; kwargs...)
    par = recovkwargs(Par, kwargs)
    Mavg(kwargs, par)
end

function transf(object::Mavg, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Mavg, X::Matrix)
    n, p = size(X)
    npoint = object.par.npoint
    kern = ImageFiltering.centered(ones(npoint) / npoint) ;
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
    savgk(nhwindow, degree, deriv)
Compute the kernel of the Savitzky-Golay filter.
* `nhwindow` : Nb. points of the half window (nhwindow >= 1) 
    --> the size of the kernel is odd (npoint = 2 * nhwindow + 1): 
    x[-nhwindow], x[-nhwindow+1], ..., x[0], ...., x[nhwindow-1], x[nhwindow].
* `degree` : Polynom order (1 <= degree <= 2 * nhwindow).
    The case "degree = 0" (simple moving average) is not allowed by the funtion.
* `deriv` : Derivation order (0 <= deriv <= degree).
    If `deriv = 0`, there is no derivation (only polynomial smoothing).

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
function savgk(nhwindow::Int, deriv::Int, degree::Int)
    @assert nhwindow >= 1 "Argument 'nhwindow' must be >= 1."
    @assert 1 <= degree <= 2 * nhwindow "Argument 'degree' must agree with: 1 <= 'degree' <= 2 * 'nhwindow'."
    @assert 0 <= deriv <= degree "Argument 'deriv' must agree with: 0 <= 'deriv' <= 'degree'."
    npoint = 2 * nhwindow + 1
    S = zeros(Int(npoint), Int(degree) + 1) ;
    u = collect(-nhwindow:nhwindow)
    @inbounds for j in 0:degree
        S[:, j + 1] .= u.^j
    end
    G = S * inv(S' * S)
    kern = factorial(deriv) * vcol(G, deriv + 1)
    (S = S, G = G, kern = kern)
end

"""
    savgol(X; npoint, degree, deriv)
    savgol!(X::Matrix; npoint, degree, deriv)
Savitzky-Golay smoothing of each row of a matrix `X`.
* `X` : X-data (n, p).
* `npoint` : Size of the filter (nb. points involved in 
    the kernel). Must be odd and >= 3. The half-window size is 
    nhwindow = (npoint - 1) / 2.
* `degree` : Polynom order (1 <= degree <= npoint - 1).
* `deriv` : Derivation order (0 <= deriv <= degree).

The smoothing is computed by convolution (with padding), with function 
imfilter of package ImageFiltering.jl. Each returned point is located on the fcenter 
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
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wlstr = names(dat.X)
wl = parse.(Float64, wlstr)

npoint = 21 ; degree = 3 ; deriv = 2 ; 
Xp = savgol(X; npoint = npoint, degree = degree, deriv = deriv) 
plotsp(Xp[1:30, :], wl).f
```
""" 
function savgol(X; kwargs...)
    par = recovkwargs(Par, kwargs)
    Savgol(kwargs, par)
end

function transf(object::Savgol, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Savgol, X::Matrix)
    npoint = object.par.npoint 
    @assert isodd(npoint) && npoint >= 3 "Argument 'npoint' must be odd and >= 3."
    n, p = size(X)
    deriv = object.par.deriv
    degree = object.par.degree
    nhwindow = Int((npoint - 1) / 2)
    kern = savgk(nhwindow, deriv, degree).kern
    zkern = ImageFiltering.centered(kern)
    out = similar(X, p)
    @inbounds for i = 1:n
        ## convolution with "replicate" padding
        imfilter!(out, vrow(X, i), reflect(zkern))
        X[i, :] .= out
    end
    ## Not faster
    #@Threads.threads for i = 1:n
    #    X[i, :] .= imfilter(vrow(X, i), reflect(zkern))
    #end
end

"""
    snv(X; centr = true, scal = true)
    snv!(X::Matrix; centr = true, scal = true)
Standard-normal-variate (SNV) transformation of each row of X-data.
* `X` : X-data.
* `centr` : Logical indicating if the centering in done.
* `scal` : Logical indicating if the scaling in done.

## Examples
```julia
using JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
wlstr = names(dat.X)
wl = parse.(Float64, wlstr)

Xp = snv(X) 
plotsp(Xp[1:30, :], wl).f
```
""" 
function snv(X; kwargs...)
    par = recovkwargs(Par, kwargs)
    Snv(kwargs, par)
end

function transf(object::Snv, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Snv, X::Matrix)
    n, p = size(X)
    centr = object.par.centr 
    scal = object.par.scal
    centr ? mu = rowmean(X) : mu = zeros(n)
    scal ? s = rowstd(X) : s = ones(n)
    # Not faster: @Threads.threads
    @inbounds for j = 1:p
        X[:, j] .= (vcol(X, j) .- mu) ./ s
    end
end


"""
    center(X)
"""
function center(X)
    xmeans = colmean(X)
    Center(xmeans)
end

function center(X, weights::Weight)
    xmeans = colmean(X, weights)
    Center(xmeans)
end

function transf(object::Center, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Center, X::Matrix)
    fcenter!(X, object.xmeans)
end

"""
    scale(X)
"""
function scale(X)
    xscales = colstd(X)
    Scale(xscales)
end

function scale(X, weights::Weight)
    xscales = colstd(X, weights)
    Scale(xscales)
end

function transf(object::Scale, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Scale, X::Matrix)
    fscale!(X, object.xscales)
end

"""
    cscale(X)
"""
function cscale(X)
    xmeans = colmean(X)
    xscales = colstd(X)
    Cscale(xmeans, xscales)
end

function cscale(X, weights::Weight)
    xmeans = colmean(X, weights)
    xscales = colstd(X, weights)
    Cscale(xmeans, xscales)
end

function transf(object::Cscale, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Cscale, X::Matrix)
    fcscale!(X, object.xmeans, object.xscales)
end
