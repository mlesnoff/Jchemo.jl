struct Kde1_1
    fm
    x
    density
end

"""
    kde1(x; npoints = 2^8, kwargs...)
Univariate kernel density estimation.
* `x` : Univariate data.
* `npoints` : Nb. points for which density is estimated (from `x`).
* `kwargs` : Optional arguments to pass in function `kde` of `KernelDensity.jl`.

The function is a wrapper (with `predict` function) of the univariate KDE 
function `kde` of package `KernelDebsity.jl`

## References 

https://github.com/JuliaStats/KernelDensity.jl

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
    Kde1_1(fm, fm.x, fm.density)
end

"""
    predict(object::Kde1, x)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `x` : Data (vector) for which predictions are computed.
""" 
function predict(object::Kde1_1, x)
    KernelDensity.pdf(object.fm, vec(x))
end

