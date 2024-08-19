"""
    dtpol(X; kwargs...)
Baseline correction of each row of X-data by polynomial linear regression.
* `X` : X-data (n, p).
Keyword arguments:
* `degree` : Polynom degree.

De-trend transformation: the function fits a baseline by polynomial regression 
for each observation and returns the residuals (= signals corrected from the baseline).

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

mod = model(dtpol; degree = 2)
fit!(mod, Xtrain)
Xptrain = transf(mod, Xtrain)
Xptest = transf(mod, Xtest)
plotsp(Xptrain, wl).f
plotsp(Xptest, wl).f

## Example on 1 spectrum
i = 2
zX = Matrix(X)[i:i, :]
mod = model(dtpol; degree = 1)
fit!(mod, zX)
zXc = transf(mod, zX)   # = corrected spectrum 
B = zX - zXc            # = estimated baseline
f, ax = plotsp(zX, wl)
lines!(wl, vec(B); color = :blue)
lines!(wl, vec(zXc); color = :black)
f
```
""" 
function dtpol(X; kwargs...)
    par = recovkw(ParDtpol, kwargs).par
    Dtpol(par)
end

""" 
    transf(object::Dtpol, X)
    transf!(object::Dtpol, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Dtpol, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Dtpol, X::Matrix)
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
    @inbounds for i = 1:n
    ## Not faster: @Threads.threads
        y = vrow(X, i)
        X[i, :] .= y - vX * A * y
    end
end

"""
    dtlo(X; kwargs...)
Baseline correction of each row of X-data by LOESS regression.
* `X` : X-data (n, p).
Keyword arguments:
* `span` : Window for neighborhood selection (level of smoothing)
    for the local fitting, typically in [0, 1] (proportion).
* `degree` : Polynomial degree for the local fitting.

De-trend transformation: The function fits a baseline by LOESS regression 
for each observation and returns the residuals (= signals corrected from the baseline).

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

mod = model(dtlo; span = .8)
fit!(mod, Xtrain)
Xptrain = transf(mod, Xtrain)
Xptest = transf(mod, Xtest)
plotsp(Xptrain, wl).f
plotsp(Xptest, wl).f

## Example on 1 spectrum
i = 2
zX = Matrix(X)[i:i, :]
mod = model(dtlo; span = .75)
fit!(mod, zX)
zXc = transf(mod, zX)   # = corrected spectrum 
B = zX - zXc            # = estimated baseline
f, ax = plotsp(zX, wl)
lines!(wl, vec(B); color = :blue)
lines!(wl, vec(zXc); color = :black)
f
```
""" 
function dtlo(X; kwargs...)
    par = recovkw(ParDtlo, kwargs).par
    Dtlo(par)
end

""" 
    transf(object::Dtlo, X)
    transf!(object::Dtlo, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Dtlo, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Dtlo, X::Matrix)
    Q = eltype(X)
    n, p = size(X)
    span = object.par.span
    degree = object.par.degree
    x = convert.(Q, collect(1:p))
    @inbounds for i = 1:n
    ## Not faster: @Threads.threads
        y = vec(vrow(X, i))
        fm = loessr(x, y; span, degree)
        X[i, :] .= y - vec(predict(fm, x).pred)
    end
end

"""
    fdif(X; kwargs...)
Finite differences (discrete derivates) for each row of X-data. 
* `X` : X-data (n, p).
Keyword arguments:
* `npoint` : Nb. points involved in the window for the 
    finite differences. The range of the window 
    (= nb. intervals of two successive colums) is npoint - 1.

The method reduces the column-dimension: 
* (n, p) --> (n, p - npoint + 1). 

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

mod = model(fdif; npoint = 2) 
fit!(mod, Xtrain)
Xptrain = transf(mod, Xtrain)
Xptest = transf(mod, Xtest)
plotsp(Xptrain).f
plotsp(Xptest).f
```
""" 
function fdif(X; kwargs...)
    par = recovkw(ParFdif, kwargs).par
    Fdif(par)
end

""" 
    transf(object::Fdif, X)
    transf!(object::Fdif, X::Matrix, M::Matrix)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
* `M` : Pre-allocated output matrix (n, p - npoint + 1).
The in-place function stores the output in `M`.
""" 
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
    interpl(X; kwargs...)
Sampling spectra by interpolation.
* `X` : Matrix (n, p) of spectra (rows).
Keyword arguments:
* `wl` : Values representing the column "names" of `X`. 
    Must be a numeric vector of length p, or an AbstractRange.
* `wlfin` : Final values (within the range of `wl`) where to interpolate
    the spectrum. Must be a numeric vector, or an AbstractRange.

The function implements a cubic spline interpolation using 
package DataInterpolations.jl.

## References
Package DAtaInterpolations.jl
https://github.com/PumasAI/DataInterpolations.jl
https://htmlpreview.github.io/?https://github.com/PumasAI/DataInterpolations.jl/blob/v2.0.0/example/DataInterpolations.html

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

wlfin = range(500, 2400, length = 10)
#wlfin = collect(range(500, 2400, length = 10))
mod = model(interpl; wl, wlfin)
fit!(mod, Xtrain)
Xptrain = transf(mod, Xtrain)
Xptest = transf(mod, Xtest)
plotsp(Xptrain).f
plotsp(Xptest).f
```
"""
function interpl(X; kwargs...)
    par = recovkw(ParInterpl, kwargs).par
    Interpl(par)
end

""" 
    transf(object::Interpl, X)
    transf!(object::Interpl, X::Matrix, M::Matrix)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
* `M` : Pre-allocated output matrix (n, p).
The in-place function stores the output in `M`.
""" 
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
    mavg(X; kwargs...)
Smoothing by moving averages of each row of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `npoint` : Nb. points involved in the window. 

The smoothing is computed by convolution with padding, 
using function imfilter of package ImageFiltering.jl. 
The centered kernel is ones(`npoint`) / `npoint`. 
Each returned point is located on the center of the kernel.

The function returns a matrix (n, p).

## References
Package ImageFiltering.jl
https://github.com/JuliaImages/ImageFiltering.jl

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

mod = model(mavg; npoint = 10) 
fit!(mod, Xtrain)
Xptrain = transf(mod, Xtrain)
Xptest = transf(mod, Xtest)
plotsp(Xptrain).f
plotsp(Xptest).f
```
""" 
function mavg(X; kwargs...)
    par = recovkw(ParMavg, kwargs).par
    Mavg(par)
end

""" 
    transf(object::Mavg, X)
    transf!(object::Mavg, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Mavg, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Mavg, X::Matrix)
    n, p = size(X)
    npoint = object.par.npoint
    kern = ImageFiltering.centered(ones(npoint) / npoint) 
    out = similar(X, p) 
    @inbounds for i = 1:n
        ImageFiltering.imfilter!(out, vrow(X, i), kern)
        X[i, :] .= out
    end
    # Not faster
    #@Threads.threads for i = 1:n
    #    X[i, :] .= imfilter(vrow(X, i), kern)
    #end
end

""" 
    savgk(nhwindow::Int, degree::Int, deriv::Int)
Compute the kernel of the Savitzky-Golay filter.
* `nhwindow` : Nb. points (>= 1) of the half window.
* `degree` : Degree of the smoothing polynom, where
   1 <= `degree` <= 2 * nhwindow.
* `deriv` : Derivation order, where 0 <= `deriv` <= degree.

The size of the kernel is odd (npoint = 2 * nhwindow + 1): 
* x[-nhwindow], x[-nhwindow+1], ..., x[0], ...., x[nhwindow-1], x[nhwindow].

If `deriv` = 0, there is no derivation (only polynomial smoothing).

The case `degree` = 0 (i.e. simple moving average) is not 
allowed by the funtion.

## References
Luo, J., Ying, K., Bai, J., 2005. Savitzky–Golay smoothing and differentiation 
filter for even number data. Signal Processing 85, 1429–1434.
https://doi.org/10.1016/j.sigpro.2005.02.002

## Examples
```julia
using Jchemo
res = savgk(21, 3, 2)
pnames(res)
res.S 
res.G 
res.kern
```
""" 
function savgk(nhwindow::Int, degree::Int, deriv::Int)
    @assert nhwindow >= 1 "Argument 'nhwindow' must be >= 1."
    @assert 1 <= degree <= 2 * nhwindow "Argument 'degree' must agree with: 1 <= 'degree' <= 2 * 'nhwindow'."
    @assert 0 <= deriv <= degree "Argument 'deriv' must agree with: 0 <= 'deriv' <= 'degree'."
    npoint = 2 * nhwindow + 1
    S = zeros(Int, npoint, degree + 1) ;
    u = collect(-nhwindow:nhwindow)
    @inbounds for j in 0:degree
        S[:, j + 1] .= u.^j
    end
    G = S * inv(S' * S)
    kern = (-1)^deriv * factorial(deriv) * vcol(G, deriv + 1) # = h_d in Luo et al. 2005 Eq.5
    (S = S, G = G, kern = kern)
end

"""
    savgol(X; kwargs...)
Savitzky-Golay derivation and smoothing of each row of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `npoint` : Size of the filter (nb. points involved in 
    the kernel). Must be odd and >= 3. The half-window size is 
    nhwindow = (`npoint` - 1) / 2.
* `degree` : Degree of the smoothing polynom.
    Must be: 1 <= `degree` <= `npoint` - 1.
* `deriv` : Derivation order. Must be: 0 <= `deriv` <= `degree`.

The smoothing is computed by convolution (with padding), using 
function imfilter of package ImageFiltering.jl. Each returned point is 
located on the center of the kernel. The kernel is computed with 
function `savgk`.

The function returns a matrix (n, p).

## References 
Luo, J., Ying, K., Bai, J., 2005. Savitzky–Golay smoothing and differentiation filter for 
even number data. Signal Processing 85, 1429–1434. https://doi.org/10.1016/j.sigpro.2005.02.002

Savitzky, A., Golay, M.J.E., 2002. Smoothing and Differentiation of Data by Simplified Least 
Squares Procedures. [WWW Document]. https://doi.org/10.1021/ac60214a047

Schafer, R.W., 2011. What Is a Savitzky-Golay Filter? [Lecture Notes]. 
IEEE Signal Processing Magazine 28, 111–117. https://doi.org/10.1109/MSP.2011.941097

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

npoint = 11 ; degree = 2 ; deriv = 2
mod = model(savgol; npoint, degree, deriv) 
fit!(mod, Xtrain)
Xptrain = transf(mod, Xtrain)
Xptest = transf(mod, Xtest)
plotsp(Xptrain).f
plotsp(Xptest).f

####### Gaussian signal 

u = -15:.1:15
n = length(u)
x = exp.(-.5 * u.^2) / sqrt(2 * pi) + .03 * randn(n)
M = 10  # half window
N = 3   # degree
deriv = 0
#deriv = 1
mod = model(savgol; npoint = 2M + 1, degree = N, deriv)
fit!(mod, x')
xp = transf(mod, x')
f, ax = plotsp(x', u; color = :blue)
lines!(ax, u, vec(xp); color = :red)
f
```
""" 
function savgol(X; kwargs...)
    par = recovkw(ParSavgol, kwargs).par
    Savgol(par)
end

""" 
    transf(object::Savgol, X)
    transf!(object::Savgol, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Savgol, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Savgol, X::Matrix)
    npoint = object.par.npoint 
    @assert isodd(npoint) && npoint >= 3 "Argument 'npoint' must be odd and >= 3."
    n, p = size(X)
    degree = object.par.degree
    nhwindow = Int((npoint - 1) / 2)
    deriv = object.par.deriv
    kern = savgk(nhwindow, degree, deriv).kern
    kernc = ImageFiltering.centered(kern)
    x = similar(X, p)
    @inbounds for i = 1:n
        ## Convolution with "replicate" padding
        ImageFiltering.imfilter!(x, vrow(X, i), reflect(kernc))
        X[i, :] .= x
        ## Alternatves not fasters (~ same)
        #ImageFiltering.imfilter!(X[i, :], vrow(X, i), reflect(kernc))
        #ImageFiltering.imfilter!(vrow(X, i), vrow(X, i), reflect(kernc))
    end
    ## Not faster
    #@Threads.threads for i = 1:n
    #    X[i, :] .= imfilter(vrow(X, i), reflect(kernc))
    #end
end

"""
    snorm(X)
Row-wise norming of X-data.
* `X` : X-data (n, p).

Each row of `X` is divide by its norm.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

mod = model(snorm) 
fit!(mod, Xtrain)
Xptrain = transf(mod, Xtrain)
Xptest = transf(mod, Xtest)
plotsp(Xptrain).f
plotsp(Xptest).f
rownorm(Xptrain)
rownorm(Xptest)
```
""" 
function snorm(X)
    Snorm()
end

""" 
    transf(object::Snorm, X)
    transf!(object::Snorm, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Snorm, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Snorm, X::Matrix)
    X ./= rownorm(X)
end

"""
    snv(X; kwargs...)
Standard-normal-variate (SNV) transformation of each row of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `centr` : Boolean indicating if the centering in done.
* `scal` : Boolean indicating if the scaling in done.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

mod = model(snv) 
#mod = model(snv; scal = false) 
fit!(mod, Xtrain)
Xptrain = transf(mod, Xtrain)
Xptest = transf(mod, Xtest)
plotsp(Xptrain).f
plotsp(Xptest).f
```
""" 
function snv(X; kwargs...)
    par = recovkw(ParSnv, kwargs).par
    Snv(par)
end

""" 
    transf(object::Snv, X)
    transf!(object::Snv, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
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

