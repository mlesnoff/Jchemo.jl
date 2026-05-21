"""
    boxcox(x; lims = (-3., 3), npoint = 1000)
Estimate the parameter of the Box-Cox power transformation to normalize a variable.
* `x` : Univariate data (n) to normalize.
Keyword arguments:
* `lims` : .
* `npoint` : .

## References
Box, G.E.P., and D.R. Cox. (1964). An Analysis of Transformations (with Discussion).
Journal of the Royal Statistical Society, Series B 26(2), 211--252.

## Examples
```julia
```
""" 
function boxcox(x; lims = (-3., 3), npoint = 1000)
    Q = eltype(x)
    lims = convert.(Q, lims)
    if any(x .<= 0)
        error("Box-Cox transformation requires strictly positive data (x > 0).")
    end
    lb = range(lims[1], lims[2]; length = npoint)
    logl = similar(x, npoint) 
    n = length(x)
    #@inbounds for i in eachindex(lb)
    @Threads.threads for i in eachindex(lb)
        vx = boxcox_transf(x, lb[i])  # not '.=' due to racing error when threading
        sigma = stdv(vx)
        J = (lb[i] - 1) * sum(log.(x))
        logl[i] = - n * log(sigma) + J
        ## Term 1 / (2 * sigma^2) * sum((vx .- meanv(vx)).^ 2) is constant
    end
    ind = argmax(logl)[1]
    (lb = lb, logl, lbopt = lb[ind], ind)
end

## Apply Box-Cox transformation to vector x for lambda = lb
function boxcox_transf(x::Vector{T}, lb::T) where T <: Union{Float32, Float64}
    lb == 0 ? log.(x) : (x.^lb .- 1) ./ lb
end



