
"""
    eposvd(D; nlv)
Pre-processing X-data by external parameter orthogonalization (EPO).
* `D` : Data (m, p) containing detrimental information.
* `nlv` : Nb. of first loadings vectors of D considered for the orthogonalization.

The objective is to remove from a dataset X (n, p) some "detrimental" 
information (e.g. humidity patterns in signals) represented by a dataset D (m, p).

EPO (Roger et al 2003) consists in orthogonalizing the row observations 
of X to the detrimental sub-space defined by the first nlv non-centered 
PCA loadings vectors of D.

Function eposvd makes a SVD factorization of D and returns 
tow matrices:
* M (p, p) : The orthogonalization matrix that can be used to correct the X-data.
* P (p, nlv) : The matrix of the considered loading vectors of D. 

The X-data corrected from the detrimental information D 
can be computed by X_corrected = X * M.

# References

Roger, J.-M., Chauchard, F., Bellon-Maurel, V., 2003. EPO-PLS external parameter 
orthogonalisation of PLS application to temperature-independent measurement 
of sugar content of intact fruits. 
Chemometrics and Intelligent Laboratory Systems 66, 191-204. 
https://doi.org/10.1016/S0169-7439(03)00051-0

Roger, J.-M., Boulet, J.-C., 2018. A review of orthogonal projections for calibration. 
Journal of Chemometrics 32, e3045. https://doi.org/10.1002/cem.3045

# Example

```julia
n = 4 ; p = 8 ; 
X = rand(n, p)
m = 3 ;
D = rand(m, p)    # Detrimental information

nlv = 2
res = eposvd(D; nlv = nlv)
res.M      # orthogonalization matrix
res.P      # detrimental directions (columns of P = loadings of D)
```

The matrix corrected from D can be computed by:
```julia
X_corr = X * res.M    
```
Rows of the corrected matrix X_corr
are orthogonal to the loadings vectors (columns of P):
```julia
X_corr * res.P 
```
""" 
function eposvd(D; nlv)
    D = ensure_mat(D)
    m, p = size(D)
    nlv = min(nlv, m, p)
    I = Diagonal(ones(p))
    if nlv == 0 
        M = I
        P = nothing
    else 
        P = svd!(D).V[:, 1:nlv]
        M = I - P * P'
    end
    (M = M, P = P)
end

"""
    detrend(X)
Linear de-trend transformation of each row of X-data. 
* `X` : X-data.

The function fits a univariate linear regression to each observation
and returns the residuals.
""" 
function detrend(X)
    M = copy(X)
    detrend!(M)
    M
end

function detrend!(X)
    n, p = size(X)
    xmean = mean(1:p)
    x = collect(1:p) 
    xc = x .- xmean
    xtx = dot(xc, xc)
    yc = similar(xc)
    @inbounds for i = 1:n
        vy = vrow(X, i)
        ymean = mean(vy)  
        yc .= vy .- ymean
        b = dot(xc, yc) / xtx
        int = ymean - xmean * b
        X[i, :] .= vy .- (int .+ x * b)
    end
end

"""
    fdif(X; f = 2)
    fdif!(M, X; f = 2)
Finite differences of each row of a matrix X. 
* `X` : X-data (n, p).
* `M` : Pre-allocated output matrix (n, p - f + 1).
* `f` : Size of the window (nb. points involved) for the finite differences.
    The range of the window (= nb. intervals of two successive colums) is f - 1.

The finite differences can be used for computing discrete derivates.
The method reduces the column-dimension: (n, p) --> (n, p - f + 1). 

The in-place function stores the output in `M`.
""" 
function fdif(X; f = 2)
    n, p = size(X)
    M = similar(X, n, p - f + 1)
    fdif!(M, X; f)
    M
end

function fdif!(M, X; f = 2)
    p = size(X, 2)
    zp = p - f + 1
    @inbounds for j = 1:zp
        M[:, j] .= vcol(X, j + f - 1) .- vcol(X, j)
    end
end

""" 
    interpl(X; w0, w, 
        fun = CubicSpline)
Sampling signals by interpolation.
* `X` : Matrix (n, p) of signals (rows).
* `w0` : Values representing the column "names" of `X`. 
    Must be a numeric vector of length p, or an AbstractRange.
* `w` : Values where to interpolate within the range of `w0`.
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
""" 
function interpl(X; w0, w, 
        fun = cubic_spline)
    X = ensure_mat(X)
    n = size(X, 1)
    q = length(w)
    zX = similar(X, n, q)
    @inbounds for i = 1:n
        itp = fun(vrow(X, i), w0)
        zX[i, :] .= itp.(w)
    end
    zX
end
# Due to conflicts with Interpolations.jl
linear_int(y, x) = DataInterpolations.LinearInterpolation(y, x)
quadratic_int(y, x) = DataInterpolations.QuadraticInterpolation(y, x)
quadratic_spline(y, x) = DataInterpolations.QuadraticSpline(y, x)
cubic_spline(y, x) = DataInterpolations.CubicSpline(y, x)

""" 
    interpl_mon(X; w0, w, 
        fun = FritschCarlsonMonotonicInterpolation)
Sampling signals by monotonic interpolation.
* `X` : Matrix (n, p) of signals (rows).
* `w0` : Values representing the column "names" of `X`. 
    Must be a numeric vector of length p, or an AbstractRange.
* `w` : Values where to interpolate within the range of `w0`.
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
function interpl_mon(X; w0, w, 
        fun = FritschCarlsonMonotonicInterpolation)
    X = ensure_mat(X)
    n = size(X, 1)
    q = length(w)
    zX = similar(X, n, q)
    @inbounds for i = 1:n
        itp = interpolate(w0, vrow(X, i), fun())
        zX[i, :] .= itp.(w)
    end
    zX
end

"""
    mavg(X; f)
Moving averages smoothing of each row of X-data.
* `X` : X-data.
* `f` : Size (nb. points involved) of the filter.

The smoothing is computed by convolution (with padding), with function 
imfilter of package ImageFiltering.jl. The centered kernel is ones(`f`) / `f`.
Each returned point is located on the center of the kernel.
""" 
function mavg(X; f)
    M = copy(X)
    mavg!(M; f)
    M
end

function mavg!(X; f)
    n, p = size(X)
    kern = ImageFiltering.centered(ones(f) / f) ;
    out = similar(X, p)
    @inbounds for i = 1:n
        imfilter!(out, vrow(X, i), kern)
        X[i, :] .= out
    end
end

"""
    mavg_runmean(X, f)
    mavg_runmean!(M, X, f)
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
""" 
function mavg_runmean(X; f)
    n, p = size(X)
    M = similar(X, n, p - f + 1)
    mavg_runmean!(M, X; f)
    M
end

function mavg_runmean!(M, X; f)
    n, zp = size(M)
    out = similar(M, zp)
    @inbounds for i = 1:n
        Jchemo.runmean!(out, vrow(X, i); f)
        M[i, :] .= out    
    end
end

## adaptation from function runmean  (V. G. Gumennyy)
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
Savitzky-Golay smoothing of each row of a matrix `X`.
* `X` : X-data (n, p).
* `f` : Size of the filter (nb. points involved in the kernel). Must be odd and >= 3.
    The half-window size is m = (f - 1) / 2.
* `pol` : Polynom order (1 <= pol <= f - 1).
* `d` : Derivation order (0 <= d <= pol).

The smoothing is computed by convolution (with padding), with function 
imfilter of package ImageFiltering.jl. Each returned point is located on the center 
of the kernel. The kernel is computed with function `savgk`.
""" 
function savgol(X; f, pol, d)
    M = copy(X)
    savgol!(M; f, pol, d)
    M
end

function savgol!(X; f, pol, d)
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
end

"""
    snv(X)
Standard-normal-variate (SNV) transformation of each row of X-data.
* `X` : X-data.

Each row of `X` is centered and scaled.
""" 
function snv(X)
    M = copy(X)
    snv!(M)
    M
end

function snv!(X) 
    p = size(X, 2)
    mu = vec(Statistics.mean(X; dims = 2))
    s = vec(Statistics.std(X; corrected = false, dims = 2))
    @inbounds for j = 1:p
        X[:, j] .= (vcol(X, j) .- mu) ./ s
    end
end


