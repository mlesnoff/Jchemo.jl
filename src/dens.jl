struct Dens3
    d
    mu
    lims
end

"""
    dens(x; npoints = 2^8, lims = nothing)
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
function dens(x; npoints = 2^8, lims = nothing)
    x = Float64.(vec(x))
    isnothing(lims) ? lims = (minimum(x), maximum(x)) : nothing
    fm = Makie.KernelDensity.kde(x; npoints = npoints,
        boundary = lims)
    d = mweight(fm.density)   # Sum(d) = 1
    mu = 1 / npoints          # Average expected density value; d should be >= mu if "dense" area
    s = (fm.x .< lims[1]) .| (fm.x .> lims[2])
    d = DataFrame(x = fm.x, d = d, out = s)
    Dens3(d, mu, lims)        
end

"""
    predict(object::Dens2, x)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `x` : Data (vector) for which predictions are computed.
""" 
function predict(object::Dens3, x)
    x = Float64.(vec(x))
    n = length(object.d.x)
    m = length(x)
    s = (x .< object.lims[1]) .| (x .> object.lims[2])
    d = object.d.d
    zd = zeros(m)
    zs = findall(s .== 0)
    zd[zs] .= vec(interpl(reshape(d, 1, n), object.d.x; 
        wlfin = x[zs]))
    d = DataFrame(x = x, d = zd, out = s)
    (d = d,)
end



