struct Kde11
    fm
end

"""
    kde1(x; npoints = 2^8, lims = nothing)
Univariate kernel density estimation.
* `x` : Data (vector) on which the density is estimated.
* `npoints` : Nb. points estimated within the range of `lims`.
* `lims` : If `nothing`, this is `[minimum(x); maximum(x)]`.

The function uses function `Makie.KernelDensity.kde`.
After estimation, the densities (`d`) are normalized to sum to 1. 

## Examples
```julia
n = 10^3 
x = randn(n)

fm = dens(x)
pnames(fm)
fm.d     # Density (normalized to sum to 1)
fm.lims  # Range of x
fm.mu    # Mean expected density value for a uniform distribution
f, ax = lines(fm.d.x, fm.d.d;
    axis = (xlabel = "x", ylabel = "Density"))
hlines!(ax, fm.mu; color = :grey, linestyle = "-")
f

xnew = [-4; -1; -3.8; 1; 4]
d = Jchemo.predict(fm, xnew).d
scatter!(ax, xnew, d.d; color = :red)
f
```
""" 
function kde1(x; npoints = 2^8, kwargs...)
    fm = KernelDensity.kde(vec(x); npoints = npoints, kwargs...)
    Kde11(fm)
end

"""
    predict(object::Kde1, x)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `x` : Data (vector) for which predictions are computed.
""" 
function predict(object::Kde11, x)
    KernelDensity.pdf(object.fm, vec(x))
end

