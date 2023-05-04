struct Kde1
    fm
    x
    density
end

"""
    kde1(x; npoints = 2^8, kwargs...)
Univariate kernel density estimation (KDE).
* `x` : Univariate data.
* `npoints` : Nb. points for which density is estimated (from `x`).
* `kwargs` : Optional arguments to pass in function `kde` of `KernelDensity.jl`.

The function is a wrapper (with `predict` function) of the univariate KDE 
function `kde` of package `KernelDebsity.jl`

## References 
https://github.com/JuliaStats/KernelDensity.jl

## Examples
```julia
using CairoMakie

n = 10^4
x = randn(n)
lims = (minimum(x), maximum(x))
#lims = (-10, 10)
m = 2^3
bw = .1
fm = kde1(x; npoints = m, 
    bandwidth = bw,    # default: Silverman's rule
    boundary = lims    # default: See KernelDensity
    ) ;
fm.x
diff(fm.x)
d = fm.density    # same scale as ':pdf' in Makie.hist
sum(d * diff(fm.x)[1])  # = 1
## Normalization to sum to 1
dtot = sum(d)
dn = d / dtot 
sum(dn)
## Standardization to a uniform distribution
## ("dense" areas > 1) 
mu_unif = mean(d)
d / mu_unif 

## Prediction
xnew = [-200; -100; -1; 0; 1; 200]
dnew = Jchemo.predict(fm, xnew).pred
dnew / dtot 
dnew / mu_unif 

n = 10^3 
x = randn(n)
lims = (minimum(x), maximum(x))
#lims = (-6, 6)
m = 2^8
bw = .1
fm = kde1(x; npoints = m, 
    bandwidth = bw,         # Default = Silverman's rule
    boundary = lims
    ) ;
f = Figure(resolution = (500, 350))
ax = Axis(f[1, 1];
    xlabel = "x", ylabel = "Density")
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
lines!(ax, fm.x, fm.density;
    color = :red)
f
```
""" 
function kde1(x; npoints = 2^8, kwargs...)
    fm = KernelDensity.kde(vec(x); npoints = npoints, kwargs...)
    Kde1(fm, fm.x, fm.density)
end

"""
    predict(object::Kde1, x)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `x` : Data (vector) for which predictions are computed.
""" 
function predict(object::Kde1, x)
    m = length(x)
    pred = KernelDensity.pdf(object.fm, vec(x))
    pred = reshape(pred, m, 1)
    (pred = pred,)
end

