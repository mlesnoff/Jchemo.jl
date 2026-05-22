"""
    boxcox(x::Vector{T}; lims::Vector{T} = [-3.; 3], npoint = 1000) where T <: Union{Float32, Float64}
Estimate the parameter of the Box-Cox power transformation.
* `x` : Univariate data (n) to normalize.
Keyword arguments:
* `lims` : Range of lambda values to consider (vector of length = 2).
* `npoint` : Number of points to evaluate in the grid.

The Box-Cox power transformation is defined as:
* lamda = 0 :  y = log.(x)
* else : = y (x.^lambda .- 1) ./ lambda

Parameter lambda (outputs `lb` and `lbopt`) is estimated by maximizing the log-likelihood of the 
transformed data under a Gaussian distribution. The estimation is performed on a grid of lambda 
values defined by `lims` and `npoint`.

## References
Box, G.E.P., and D.R. Cox. (1964). An Analysis of Transformations (with Discussion).
Journal of the Royal Statistical Society, Series B 26(2), 211--252.

## Examples
```julia
using Jchemo, Distributions, CairoMakie
using Random

n = 1000
distr = Chisq(8) ; x = rand(distr, n)
#x = abs2.(randn(MersenneTwister(42), 1000))
f = Figure(size = (500, 250))
hist(f[1, 1], x; bins = 50, axis = (xlabel = "x", ylabel = "Nb. obs.", title = "Original data"))
mu = meanv(x)
sigma = stdv(x)
qqplot(f[1, 2], Normal(mu, sigma), x; qqline = :identity, axis = (xlabel = "Normal quantiles", ylabel = "Sample quantiles"))
f

res = boxcox(x)
@names res 
@show res.lbopt
f = Figure(size = (500, 300))
ax = Axis(f[1, 1]; title = "Box-Cox log-likelihood", xlabel = "Lambda", ylabel = "log-L")
lines!(ax, res.lb, res.logl)
vlines!(ax, res.lb[res.ind]; color = :red, linestyle = :dash)
f

## Transformed data
vx = boxcox_transf(x, res.lbopt)
f = Figure(size = (600, 250))
ax1 = Axis(f[1, 1]; xlabel = "Original x", ylabel = "Nb. obs.")
hist!(ax1, x; bins = 50)
ax2 = Axis(f[1, 2]; xlabel = "Transformed x", ylabel = "Nb. obs.")
hist!(ax2, vx; bins = 50)
f
f = Figure(size = (500, 250))
hist(f[1, 1], vx; bins = 50, axis = (xlabel = "x", ylabel = "Nb. obs.", title = "Transformed data"))
mu = meanv(vx) ; sigma = stdv(vx)
qqplot(f[1, 2], Normal(mu, sigma), vx; qqline = :identity, axis = (xlabel = "Normal quantiles", ylabel = "Sample quantiles"))
f
```
""" 
function boxcox(x::Vector{T}; lims::Vector{T} = [-3.; 3], npoint::Int = 1000) where T <: Union{Float32, Float64}
    if any(x .<= 0)
        error("Box-Cox transformation requires strictly positive data (x > 0).")
    end
    lb = range(lims[1], lims[2]; length = npoint)
    logl = similar(x, npoint) 
    n = length(x)
    #@inbounds for i in eachindex(lb)
    @Threads.threads for i in eachindex(lb)
        vx = boxcox_transf(x, lb[i])  # inplace not used due to racing error when threading
        sigma = stdv(vx)
        J = (lb[i] - 1) * sum(log.(x))
        logl[i] = - n * log(sigma) + J
        ## Term 1 / (2 * sigma^2) * sum((vx .- meanv(vx)).^ 2) is constant
    end
    ind = argmax(logl)[1]
    (lb = lb, logl, lbopt = lb[ind], ind)
end





"""
    boxcox_transf(x::Vector{T}, lb::T) where T <: Union{Float32, Float64}
    boxcox_transf!(x::Vector{T}, lb::T) where T <: Union{Float32, Float64}
Apply Box-Cox power transformation to a variable
* `x` : Univariate data (n) to normalize.
* `lb` : Box-Cox parameter (lambda).

The Box-Cox power transformation is defined as:
* lamda = 0 :  y = log.(x)
* else : = y (x.^lambda .- 1) ./ lambda

See function `boxcox` for examples.

## References
Box, G.E.P., and D.R. Cox. (1964). An Analysis of Transformations (with Discussion).
Journal of the Royal Statistical Society, Series B 26(2), 211--252.
""" 
function boxcox_transf(x::Vector{T}, lb::T) where T <: Union{Float32, Float64}
    vx = copy(x)
    boxcox_transf!(vx, lb)
    vx
end
function boxcox_transf!(x::Vector{T}, lb::T) where T <: Union{Float32, Float64}
    lb == 0 ? x .= log.(x) : x .= (x.^lb .- 1) ./ lb
end


