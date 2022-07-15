struct Dens
    x
    d
    mu
    lims
end

"""
    dens(x; npoints = 2^8, lims = nothing)
Univariate kernel density estimation.
* `x` : Data (vector) on which the density is estimated.
* `npoints` : Nb. points estimated within the range of `lims`.
* `lims` : If `nothing`, this is (minimum(`x`), maximum(`x`)).

The function uses function `Makie.KernelDensity.kde`.
The estimated densities are normalized to sum to 1. 

## Examples
```julia
n = 10^3 
x = randn(n)

res = dens(x)
pnames(res)
res.lims  # Range of x
res.mu    # Average expected density value for a uniform distribution
f, ax = lines(res.x, res.d;
    axis = (xlabel = "x", ylabel = "Density"))
hlines!(ax, res.mu; color = :grey, linestyle = "-")
f

xnew = [-4; -1; -3.8; 1; 4]
d = Jchemo.predict(res, xnew)
scatter!(ax, xnew, d; color = :red)
f
```
""" 
function dens(x; npoints = 2^8, lims = nothing)
    zx = Float64.(vec(x))
    isnothing(lims) ? lims = (minimum(zx), maximum(zx)) : nothing
    vfm = Makie.KernelDensity.kde(zx; npoints = npoints,
        boundary = lims)
    d = mweight(vfm.density)  # Sum(d) = 1
    mu = 1 / npoints          # Average expected density value
    Dens(vfm.x, d, mu, lims)  # d should be >= mu if "dense" area      
end

"""
    predict(object::Dens, x)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `x` : Data (vector) for which predictions are computed.
""" 
function predict(object::Dens, x)
    x = vec(x)
    n = length(object.x)
    m = length(x)
    s = (x .< object.lims[1]) .| (x .> object.lims[2])
    zs = findall(s .== 0)
    d = zeros(m)
    d[zs] .= vec(interpl(reshape(object.d, 1, n), object.x; wlfin = x[zs]))
    d
end






